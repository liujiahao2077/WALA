# Copyright 2026 WALA authors. All rights reserved.
# Licensed under the MIT License.

"""WALA policy with semantic-geometric latent action supervision."""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from wala.model.tools import FRAMEWORK_REGISTRY
from wala.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from wala.model.framework.base_framework import baseframework
from wala.model.framework.share_tools import merge_framework_config
from wala.model.modules.action_model import get_action_model
from wala.model.modules.vlm import get_vlm_model
from wala.training.trainer_utils.trainer_tools import resize_images

from wala.model.modules.vision_encoder import get_dino_model


from wala.model.modules.latent_action_model import (
    ContinuousTransitionBottleneckTokenizer,
    LatentActionToTransitionTokens,
)


@dataclass
class WALAPolicyDefaultConfig:
    """Default parameters for the WALA policy."""

    name: str = "WALA"

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./checkpoints/qwen/Qwen3-VL-4B-Instruct",
        "attn_implementation": "sdpa",
    })

    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "MLP",
        "action_dim": 14,
        "state_dim": 14,
        "action_hidden_dim": 2560,
        "condition_dim": 1024,
        "future_action_window_size": 49,
        "past_action_window_size": 0,
    })

    dino: dict = field(default_factory=lambda: {
        "dino_backbone": "./checkpoints/dinov3/dinov3-vitb16-pretrain-lvd1689m",
    })

    transition_bottleneck: dict = field(default_factory=lambda: {
        "training_mode": "vla",
        "dino_dim": 768,
        "num_transition_tokens": 32,
        "hidden_dim": 1024,
        "encoder_layers": 8,
        "decoder_layers": 4,
        "latent_resampler_layers": 1,
        "num_heads": 8,
        "max_future_steps": 64,
        "max_patches": 1024,
        "max_action_tokens": 256,

        "use_depth": False,
        "depth_loss_weight": 1.0,
        "depth_fusion_scale": 1.0,
        "depth_anything_encoder": "vitl",
        "depth_anything_ckpt": "./checkpoints/depth_anything/depth_anything_v2_vitl.pth",
        "depth_anything_input_size": 224,
        "depth_patch_size": 16,

        "smooth_l1_weight": 1.0,
        "cosine_weight": 0.1,
        "composition_weight": 0,
        "reencode_weight": 0,
        "content_invariance_weight": 0,
        "content_aug_scale_std": 0.05,
        "content_aug_bias_std": 0.02,
        "min_composition_steps": 2,

        "use_transition_alignment": False,
        "transition_alignment_loss_weight": 0.01,
        "transition_alignment_l1_weight": 0.1,

        "use_transition_decode": True,
        "transition_decode_loss_weight": 0.1,
        "transition_decode_cosine_weight": 0.1,
        "transition_decode_use_all_future": True,
        "depth_decoder_loss_weight": 0.5,
        "depth_gradient_loss_weight": 0.2,
        "depth_decoder_use_all_future": True,

        "freeze_encoder_in_vla": True,
        "train_decoder_in_vla": True,
    })

    obs_image_size: Optional[list] = None
    vla_state_dropout_prob: float = 0.0

@FRAMEWORK_REGISTRY.register("WALA")
class WALAPolicy(baseframework):
    """WALA policy built on a vision-language backbone and an MLP action head."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = merge_framework_config(WALAPolicyDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.action_model_type = str(
            self.config.framework.action_model.action_model_type
        ).strip()
        if self.action_model_type != "MLP":
            raise ValueError(
                f"Unsupported action_model_type={self.action_model_type!r}; expected 'MLP'."
            )

        act_cfg = self.config.framework.action_model
        dino_cfg = self.config.framework.dino
        ft_cfg = self.config.framework.transition_bottleneck

        self.future_action_window_size = act_cfg.future_action_window_size
        self.past_action_window_size = act_cfg.past_action_window_size
        self.chunk_len = 1 + self.future_action_window_size

        self.action_token = "🔍"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer(
            "🔍", add_special_tokens=False
        )["input_ids"][0]

        self.num_transition_tokens = getattr(
            ft_cfg,
            "num_transition_tokens",
            getattr(ft_cfg, "num_future_tokens", self.chunk_len),
        )
        self.dino_feature_dim = getattr(ft_cfg, "dino_dim", 768)
        self.transition_hidden_dim = getattr(ft_cfg, "hidden_dim", self.dino_feature_dim)

        self.l1_loss = nn.L1Loss()

        # Frozen DINO feature extractor / target provider.
        self.dino_encoder = get_dino_model(backone_name=dino_cfg.dino_backbone)
        for param in self.dino_encoder.parameters():
            param.requires_grad = False

        # Optional Depth Anything V2 depth-map generator.
        self.use_depth_lam = getattr(ft_cfg, "use_depth", False)
        self.depth_anything = None
        if self.use_depth_lam:
            _DEPTH_ANYTHING_ROOT = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "third_party",
                    "Depth-Anything-V2",
                )
            )

            if _DEPTH_ANYTHING_ROOT not in sys.path:
                sys.path.insert(0, _DEPTH_ANYTHING_ROOT)

            from depth_anything_v2.dpt import DepthAnythingV2

            da_encoder = getattr(ft_cfg, "depth_anything_encoder", "vitl")
            model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
                "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
            }
            if da_encoder not in model_configs:
                raise ValueError(f"Unsupported depth_anything_encoder={da_encoder!r}.")

            self.depth_anything = DepthAnythingV2(**model_configs[da_encoder])
            ckpt = getattr(ft_cfg, "depth_anything_ckpt", None)
            if ckpt is None or ckpt == "":
                ckpt = os.path.join(
                    _DEPTH_ANYTHING_ROOT,
                    "checkpoints",
                    f"depth_anything_v2_{da_encoder}.pth",
                )
            elif not os.path.isabs(ckpt):
                ckpt = os.path.abspath(ckpt)
            self.depth_anything.load_state_dict(torch.load(ckpt, map_location="cpu"))
            self.depth_anything.eval()
            for p in self.depth_anything.parameters():
                p.requires_grad = False

        # Continuous Transition Bottleneck: future-to-transition tokenizer.
        self.transition_bottleneck = ContinuousTransitionBottleneckTokenizer(
            dino_dim=self.dino_feature_dim,
            hidden_dim=self.transition_hidden_dim,
            num_transition_tokens=self.num_transition_tokens,
            encoder_layers=getattr(ft_cfg, "encoder_layers", 8),
            decoder_layers=getattr(ft_cfg, "decoder_layers", 4),
            num_heads=ft_cfg.num_heads,
            max_future_steps=ft_cfg.max_future_steps,
            max_patches=ft_cfg.max_patches,
            smooth_l1_weight=ft_cfg.smooth_l1_weight,
            cosine_weight=ft_cfg.cosine_weight,
            composition_weight=getattr(ft_cfg, "composition_weight", 0),
            reencode_weight=getattr(ft_cfg, "reencode_weight", 0),
            content_invariance_weight=getattr(ft_cfg, "content_invariance_weight", 0),
            content_aug_scale_std=getattr(ft_cfg, "content_aug_scale_std", 0.05),
            content_aug_bias_std=getattr(ft_cfg, "content_aug_bias_std", 0.02),
            min_composition_steps=getattr(ft_cfg, "min_composition_steps", 2),
            use_depth=getattr(ft_cfg, "use_depth", False),
            depth_loss_weight=getattr(ft_cfg, "depth_loss_weight", 1.0),
            depth_gradient_loss_weight=getattr(ft_cfg, "depth_gradient_loss_weight", 0.2),
            depth_fusion_scale=getattr(ft_cfg, "depth_fusion_scale", 1.0),
            depth_patch_size=getattr(ft_cfg, "depth_patch_size", 16),
        )

        # Project / resample VLA action-token hidden states into transition-token space.
        self.latent_action_to_transition_tokens_for_encoder = LatentActionToTransitionTokens(
            action_dim=act_cfg.action_hidden_dim,
            transition_dim=self.transition_hidden_dim,
            num_transition_tokens=self.num_transition_tokens,
            num_layers=getattr(ft_cfg, "latent_resampler_layers", 1),
            num_heads=ft_cfg.num_heads,
            max_action_tokens=getattr(ft_cfg, "max_action_tokens", 256),
        )

        self.latent_action_to_transition_tokens_for_decoder = LatentActionToTransitionTokens(
            action_dim=act_cfg.action_hidden_dim,
            transition_dim=self.transition_hidden_dim,
            num_transition_tokens=self.num_transition_tokens,
            num_layers=getattr(ft_cfg, "latent_resampler_layers", 1),
            num_heads=ft_cfg.num_heads,
            max_action_tokens=getattr(ft_cfg, "max_action_tokens", 256),
        )

        # Avoid unused-parameter issues under DDP by freezing modules that are not
        # supposed to be optimized in each training stage.
        _ft_mode = getattr(ft_cfg, "training_mode", "vla")
        if _ft_mode in {"pretrain", "transition_bottleneck_pretrain", "transition_pretrain"}:
            # Pretrain only the transition bottleneck.
            for module in [
                self.qwen_vl_interface,
                self.action_model,
                self.latent_action_to_transition_tokens_for_encoder,
                self.latent_action_to_transition_tokens_for_decoder,
            ]:
                for param in module.parameters():
                    param.requires_grad = False
        else:
            if getattr(ft_cfg, "freeze_encoder_in_vla", True):
                self._set_transition_encoder_requires_grad(False)

            if not getattr(ft_cfg, "train_decoder_in_vla", True):
                self._set_transition_decoder_requires_grad(False)
                self._set_transition_depth_decoder_requires_grad(False)

    # ------------------------------------------------------------------
    # Parameter freezing helpers
    # ------------------------------------------------------------------
    def _set_transition_encoder_requires_grad(self, requires_grad: bool) -> None:
        """
        Freeze/unfreeze the target encoder and its shared representation stems.

        time_embed, patch_embed, and depth_current_patch_proj are also used by
        decoders, but are kept on the frozen side so the pretrained target-token
        coordinate system remains fixed during VLA training.
        """
        encoder_modules = [
            self.transition_bottleneck.current_input_proj,
            self.transition_bottleneck.delta_input_proj,
            self.transition_bottleneck.encoder,
        ]
        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = requires_grad

        encoder_params = [
            self.transition_bottleneck.transition_queries,
            self.transition_bottleneck.transition_token_embed,
            self.transition_bottleneck.current_type_embed,
            self.transition_bottleneck.delta_type_embed,
            self.transition_bottleneck.time_embed,
            self.transition_bottleneck.patch_embed,
        ]
        for param in encoder_params:
            param.requires_grad = requires_grad

        if getattr(self.transition_bottleneck, "use_depth", False):
            depth_encoder_modules = [
                "depth_current_patch_proj",
                "depth_delta_patch_proj",
                "depth_encoder",
                "rgb_depth_cross_fuse",
                "rgb_depth_fuse",
            ]
            for name in depth_encoder_modules:
                module = getattr(self.transition_bottleneck, name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = requires_grad

            depth_encoder_params = [
                "depth_transition_queries",
                "depth_transition_token_embed",
                "depth_current_type_embed",
                "depth_delta_type_embed",
            ]
            for name in depth_encoder_params:
                param = getattr(self.transition_bottleneck, name, None)
                if param is not None:
                    param.requires_grad = requires_grad

    def _set_transition_decoder_requires_grad(self, requires_grad: bool) -> None:
        """Freeze/unfreeze RGB decoder-only modules."""
        decoder_modules = [
            self.transition_bottleneck.horizon_decoder,
            self.transition_bottleneck.current_patch_proj,
            self.transition_bottleneck.transition_film,
            self.transition_bottleneck.delta_head,
        ]
        for module in decoder_modules:
            for param in module.parameters():
                param.requires_grad = requires_grad

        decoder_params = [
            self.transition_bottleneck.horizon_query,
        ]
        for param in decoder_params:
            param.requires_grad = requires_grad

    def _set_transition_depth_decoder_requires_grad(self, requires_grad: bool) -> None:
        """Freeze/unfreeze depth decoder-only modules."""
        if not getattr(self.transition_bottleneck, "use_depth", False):
            return

        depth_decoder_modules = [
            "depth_spatial_decoder",
            "depth_decode_patch_proj",
            "depth_patch_film",
            "depth_patch_pixel_head",
            "depth_dense_refine",
        ]
        for name in depth_decoder_modules:
            module = getattr(self.transition_bottleneck, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = requires_grad

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _build_training_visual_inputs(self, examples: List[dict]):
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example.get("action", None) for example in examples]

        def safe_get_cur(imgs, idx):
            """Safely extract current frame [-1] for a view."""
            try:
                return imgs[idx][0][-1]
            except Exception:
                return None

        def safe_get_seq(imgs, view_idx, seq_idx):
            """Safely extract a sequence: seq_idx=0 past/current, seq_idx=1 future."""
            try:
                seq = imgs[view_idx][seq_idx]
                return seq if isinstance(seq, list) else [seq]
            except Exception:
                return None

        valid_pc_len, valid_f_len = 1, 1
        for imgs in batch_images:
            pc = safe_get_seq(imgs, 0, 0) or safe_get_seq(imgs, 1, 0) or safe_get_seq(imgs, 2, 0)
            if pc is not None:
                valid_pc_len = len(pc)
            fut = safe_get_seq(imgs, 0, 1) or safe_get_seq(imgs, 1, 1) or safe_get_seq(imgs, 2, 1)
            if fut is not None:
                valid_f_len = len(fut)
            if pc is not None and fut is not None:
                break
        
        processed_batch_images = []
        head_past_and_current_images = []
        head_future_images = []

        for images in batch_images:
            h_cur = safe_get_cur(images, 0)
            l_cur = safe_get_cur(images, 1)
            r_cur = safe_get_cur(images, 2)

            ref_img = h_cur or l_cur or r_cur
            if ref_img is not None:
                blank_img = Image.new(ref_img.mode, ref_img.size, color=0)
            else:
                blank_img = Image.new("RGB", (224, 224), color=0)

            processed_batch_images.append([
                h_cur or blank_img,
                l_cur or blank_img,
                r_cur or blank_img,
            ])

            head_past_and_current_images.append(safe_get_seq(images, 0, 0) or [blank_img] * valid_pc_len)
            head_future_images.append(safe_get_seq(images, 0, 1) or [blank_img] * valid_f_len)

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            processed_batch_images = resize_images(processed_batch_images, target_size=train_obs_image_size)
            head_past_and_current_images = resize_images(head_past_and_current_images, train_obs_image_size)
            head_future_images = resize_images(head_future_images, train_obs_image_size)

        return processed_batch_images, instructions, actions, head_past_and_current_images, head_future_images

    def _encode_dino_current_and_future(
        self,
        head_past_and_current_images: List[list],
        head_future_images: List[list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = len(head_past_and_current_images)

        dino_input_of_past_and_current = self.dino_encoder.prepare_dino_input(head_past_and_current_images)
        dino_input_of_future = self.dino_encoder.prepare_dino_input(head_future_images)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            dino_feat_pc = self.dino_encoder(dino_input_of_past_and_current)
            dino_feat_future = self.dino_encoder(dino_input_of_future)

        dino_feat_pc = dino_feat_pc.float()
        dino_feat_future = dino_feat_future.float()

        _, L_dino, H_dino = dino_feat_pc.shape
        N_pc = dino_feat_pc.shape[0] // B
        assert N_pc >= 1, f"Failed to split DINO current-frame features, N_pc={N_pc}."

        dino_feat_pc = dino_feat_pc.view(B, N_pc, L_dino, H_dino)
        dino_feature_current = dino_feat_pc[:, -1, :, :]

        _, L_future, H_future = dino_feat_future.shape
        N_future = dino_feat_future.shape[0] // B
        assert N_future >= 1, f"Failed to split DINO future-frame features, N_future={N_future}."
        assert L_future == L_dino and H_future == H_dino, (
            f"DINO current/future feature dimensions mismatch: current=({L_dino}, {H_dino}), "
            f"future=({L_future}, {H_future})"
        )

        dino_feature_future = dino_feat_future.view(B, N_future, L_future, H_future)
        return dino_feature_current, dino_feature_future

    # ------------------------------------------------------------------
    # Depth extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _infer_depth_anything_pil(self, img: Image.Image) -> torch.Tensor:
        """Run Depth Anything on a PIL image and return a depth map tensor [H, W]."""
        if self.depth_anything is None:
            raise RuntimeError("Depth Anything is not initialized. Set transition_bottleneck.use_depth=True.")

        ft_cfg = self.config.framework.transition_bottleneck
        input_size = getattr(ft_cfg, "depth_anything_input_size", 224)

        if not isinstance(img, Image.Image):
            img = to_pil_preserve(img)

        rgb = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Do not call infer_image() directly: its image2tensor() chooses a generic
        # cuda device, which can mismatch the current DDP rank. We move the input
        # to the actual Depth Anything parameter device explicitly.
        image, (h, w) = self.depth_anything.image2tensor(bgr, input_size)
        da_device = next(self.depth_anything.parameters()).device
        image = image.to(device=da_device, dtype=torch.float32)

        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if da_device.type == "cuda" else nullcontext()
        with ctx:
            depth = self.depth_anything.forward(image)

        depth = F.interpolate(
            depth[:, None].float(),
            (h, w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

        return depth.detach().float()

    @torch.no_grad()
    def _depth_maps_from_image_sequences(self, image_sequences: List[list]) -> torch.Tensor:
        """
        Convert PIL image sequences to Depth Anything depth maps.

        image_sequences: List length B, each item is List[PIL.Image] length N
        return: [B, N, H, W], torch.float32 on current model device
        """
        if self.depth_anything is None:
            raise RuntimeError("Depth Anything is not initialized. Set transition_bottleneck.use_depth=True.")

        B = len(image_sequences)
        if B == 0:
            raise ValueError("image_sequences is empty.")
        N = len(image_sequences[0])

        all_depths = []
        for seq in image_sequences:
            if len(seq) != N:
                raise ValueError("All image sequences should have the same length after padding/resizing.")

            seq_depths = []
            for img in seq:
                seq_depths.append(self._infer_depth_anything_pil(img).cpu())
            all_depths.append(torch.stack(seq_depths, dim=0))

        depth = torch.stack(all_depths, dim=0)
        device = next(self.parameters()).device
        return depth.to(device=device, dtype=torch.float32)

    def _encode_depth_current_and_future(
        self,
        head_past_and_current_images: List[list],
        head_future_images: List[list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
          current_depth: [B, H, W]
          future_depth:  [B, T, H, W]
        """
        depth_pc = self._depth_maps_from_image_sequences(head_past_and_current_images)
        depth_future = self._depth_maps_from_image_sequences(head_future_images)
        current_depth = depth_pc[:, -1, :, :]
        return current_depth, depth_future

    # ------------------------------------------------------------------
    # Text/state formatting
    # ------------------------------------------------------------------
    def _format_state_as_text(self, state) -> str:
        if state is None:
            return ""

        if torch.is_tensor(state):
            arr = state.detach().cpu().float().numpy()
        else:
            arr = np.asarray(state, dtype=np.float32)

        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr[-1].reshape(-1)
        else:
            arr = arr.reshape(-1)

        arr = np.clip(arr, -1.0, 1.0)
        q = np.round((arr + 1.0) * 255 / 2).astype(np.uint8)
        items = [str(int(v)) for v in q]
        return " The current discretized robot proprioceptive state is: [" + ",".join(items) + "]."

    # ------------------------------------------------------------------
    # Forward routers
    # ------------------------------------------------------------------
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        mode = kwargs.get("training_mode", kwargs.get("mode", None))
        if mode is None:
            mode = getattr(self.config.framework.transition_bottleneck, "training_mode", "vla")

        if mode in {"pretrain", "transition_bottleneck_pretrain", "transition_pretrain"}:
            return self.forward_transition_bottleneck_pretrain(examples=examples, **kwargs)
        return self.forward_vla(examples=examples, **kwargs)

    def forward_transition_bottleneck_pretrain(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Pretrain the continuous transition bottleneck.
        This path does not run the vision-language backbone or action prediction.
        """
        _, _, _, head_past_and_current_images, head_future_images = self._build_training_visual_inputs(examples)

        dino_feature_current, dino_feature_future = self._encode_dino_current_and_future(
            head_past_and_current_images=head_past_and_current_images,
            head_future_images=head_future_images,
        )

        current_depth = None
        future_depth = None
        if getattr(self.config.framework.transition_bottleneck, "use_depth", False):
            current_depth, future_depth = self._encode_depth_current_and_future(
                head_past_and_current_images=head_past_and_current_images,
                head_future_images=head_future_images,
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokenizer_out = self.transition_bottleneck(
                dino_feature_current.float(),
                dino_feature_future.float(),
                current_depth=current_depth,
                future_depth=future_depth,
            )

        total_loss = tokenizer_out["loss"]
        zero = total_loss.detach() * 0.0

        return {
            "transition_bottleneck_loss": tokenizer_out["loss"],
            "transition_bottleneck_recon_loss": tokenizer_out["recon_loss"],
            "transition_bottleneck_cosine_loss": tokenizer_out["cosine_loss"],
            "transition_bottleneck_composition_loss": tokenizer_out["composition_loss"],
            "transition_bottleneck_reencode_loss": tokenizer_out["reencode_loss"],
            "transition_bottleneck_content_invariance_loss": tokenizer_out["content_invariance_loss"],
            "transition_bottleneck_depth_recon_loss": tokenizer_out.get("depth_recon_loss", zero),
            "total_loss": total_loss,
        }

    def forward_vla(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images, instructions, actions, head_past_and_current_images, head_future_images = \
            self._build_training_visual_inputs(examples)

        states = [example.get("state", None) for example in examples]
        state_dropout_prob = float(getattr(self.config.framework, "vla_state_dropout_prob", 0.0) or 0.0)
        if not 0.0 <= state_dropout_prob <= 1.0:
            raise ValueError(f"vla_state_dropout_prob must be in [0, 1], got {state_dropout_prob}")
        if self.training and state_dropout_prob > 0.0:
            states_for_prompt = [
                None if state is not None and torch.rand(()).item() < state_dropout_prob else state
                for state in states
            ]
        else:
            states_for_prompt = states
        state_texts = [self._format_state_as_text(state) for state in states_for_prompt]

        valid_indices = [i for i, a in enumerate(actions) if a is not None]

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [
            instruction + state_text + prompt_suffix
            for instruction, state_text in zip(instructions, state_texts)
        ]

        dino_feature_current, dino_feature_future = self._encode_dino_current_and_future(
            head_past_and_current_images=head_past_and_current_images,
            head_future_images=head_future_images,
        )

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            input_ids = qwen_inputs.get("input_ids", None)

            latent_action = self._gather_token_embeddings(
                last_hidden, input_ids, self.action_token_id, self.chunk_len
            )

            transition_alignment_loss = 0.0 * latent_action.sum()
            transition_alignment_cosine_loss = 0.0 * latent_action.sum()
            transition_alignment_l1_loss = 0.0 * latent_action.sum()
            transition_depth_decode_loss = 0.0 * latent_action.sum()
            transition_depth_decode_pixel_loss = 0.0 * latent_action.sum()
            transition_depth_decode_grad_loss = 0.0 * latent_action.sum()

            current_depth = None
            future_depth = None
            if getattr(self.config.framework.transition_bottleneck, "use_depth", False):
                current_depth, future_depth = self._encode_depth_current_and_future(
                    head_past_and_current_images=head_past_and_current_images,
                    head_future_images=head_future_images,
                )

            if getattr(self.config.framework.transition_bottleneck, "use_transition_alignment", True):
                with torch.no_grad():
                    target_transition_tokens = self.transition_bottleneck.encode(
                        dino_feature_current.float(),
                        dino_feature_future.float(),
                        current_depth=current_depth,
                        future_depth=future_depth,
                    )
                pred_transition_tokens_for_encoder = self.latent_action_to_transition_tokens_for_encoder(latent_action)
                target_transition_tokens = target_transition_tokens.detach().float()

                pred_norm = F.normalize(pred_transition_tokens_for_encoder.float(), dim=-1)
                target_norm = F.normalize(target_transition_tokens, dim=-1)
                transition_alignment_cosine_loss = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()

                pred_ln = F.layer_norm(
                    pred_transition_tokens_for_encoder.float(),
                    pred_transition_tokens_for_encoder.shape[-1:],
                )
                target_ln = F.layer_norm(target_transition_tokens, target_transition_tokens.shape[-1:])
                transition_alignment_l1_loss = F.smooth_l1_loss(pred_ln, target_ln)

                alignment_l1_w = getattr(
                    self.config.framework.transition_bottleneck,
                    "transition_alignment_l1_weight",
                    0.1,
                )
                transition_alignment_loss = transition_alignment_cosine_loss + alignment_l1_w * transition_alignment_l1_loss

            transition_decode_loss = 0.0 * latent_action.sum()
            transition_decode_l1_loss = 0.0 * latent_action.sum()
            transition_decode_cosine_loss = 0.0 * latent_action.sum()

            if getattr(self.config.framework.transition_bottleneck, "use_transition_decode", True):
                target_delta = (
                    dino_feature_future.float()
                    - dino_feature_current[:, None, :, :].float()
                ).detach()

                pred_transition_tokens_for_decoder = self.latent_action_to_transition_tokens_for_decoder(latent_action)

                pred_delta = self.transition_bottleneck.decode_delta(
                    dino_feature_current.float(),
                    pred_transition_tokens_for_decoder.float(),
                    num_future_steps=dino_feature_future.shape[1],
                )

                if getattr(self.config.framework.transition_bottleneck, "transition_decode_use_all_future", True):
                    pred_delta_for_loss = pred_delta.float()
                    target_delta_for_loss = target_delta.float()
                else:
                    pred_delta_for_loss = pred_delta[:, -1, :, :].float()
                    target_delta_for_loss = target_delta[:, -1, :, :].float()

                transition_decode_l1_loss = F.l1_loss(pred_delta_for_loss, target_delta_for_loss)

                pred_delta_norm = F.normalize(pred_delta_for_loss.float(), dim=-1)
                target_delta_norm = F.normalize(target_delta_for_loss.float(), dim=-1)
                transition_decode_cosine_loss = 1.0 - (pred_delta_norm * target_delta_norm).sum(dim=-1).mean()

                decode_cos_w = getattr(
                    self.config.framework.transition_bottleneck,
                    "transition_decode_cosine_weight",
                    0.1,
                )

                # Also supervise dense depth reconstruction in VLA decoder stage.
                if getattr(self.config.framework.transition_bottleneck, "use_depth", False):
                    B, T, L, _ = dino_feature_future.shape
                    depth_patch_hw = self.transition_bottleneck._infer_square_patch_hw(L)

                    current_depth_norm, future_depth_norm = self.transition_bottleneck._normalize_depth_clip(
                        current_depth=current_depth.to(device=dino_feature_current.device),
                        future_depth=future_depth.to(device=dino_feature_current.device),
                    )

                    target_depth_delta = (
                        future_depth_norm - current_depth_norm[:, None, :, :]
                    ).detach()

                    pred_depth_delta = self.transition_bottleneck._decode_dense_depth_delta(
                        current_depth=current_depth_norm,
                        transition_tokens=pred_transition_tokens_for_decoder.float(),
                        num_future_steps=T,
                        patch_hw=depth_patch_hw,
                    )

                    if getattr(self.config.framework.transition_bottleneck, "depth_decoder_use_all_future", True):
                        pred_depth_delta_for_loss = pred_depth_delta.float()
                        target_depth_delta_for_loss = target_depth_delta.float()
                    else:
                        pred_depth_delta_for_loss = pred_depth_delta[:, -1, :, :].float()
                        target_depth_delta_for_loss = target_depth_delta[:, -1, :, :].float()

                    transition_depth_decode_pixel_loss = F.smooth_l1_loss(
                        pred_depth_delta_for_loss,
                        target_depth_delta_for_loss,
                    )
                    transition_depth_decode_grad_loss = self.transition_bottleneck._depth_gradient_l1_loss(
                        pred_depth_delta_for_loss,
                        target_depth_delta_for_loss,
                    )
                    depth_grad_w = getattr(
                        self.config.framework.transition_bottleneck,
                        "depth_gradient_loss_weight",
                        0.2,
                    )
                    transition_depth_decode_loss = (
                        transition_depth_decode_pixel_loss
                        + depth_grad_w * transition_depth_decode_grad_loss
                    )

                depth_decode_w = getattr(self.config.framework.transition_bottleneck, "depth_decoder_loss_weight", 0.5)

                transition_decode_loss = (
                    transition_decode_l1_loss
                    + decode_cos_w * transition_decode_cosine_loss
                    + depth_decode_w * transition_depth_decode_loss
                )

            transition_alignment_w = getattr(
                self.config.framework.transition_bottleneck,
                "transition_alignment_loss_weight",
                0.01,
            )
            transition_decode_w = getattr(
                self.config.framework.transition_bottleneck,
                "transition_decode_loss_weight",
                0.1,
            )

        with torch.autocast("cuda", dtype=torch.float32):
            if len(valid_indices) > 0:
                latent_action = latent_action.float()
                
                valid_actions = [actions[i] for i in valid_indices]
                valid_latent_action = latent_action[valid_indices]
                

                actions_tensor = torch.tensor(
                    np.array(valid_actions), device=latent_action.device, dtype=latent_action.dtype
                )
                assert actions_tensor.shape[1] >= self.chunk_len, (
                    f"Action sequence length ({actions_tensor.shape[1]}) is shorter than "
                    f"chunk_len ({self.chunk_len}). Please check your config and data alignment."
                )
                actions_target = actions_tensor[:, -self.chunk_len:, :]
                action_loss = self.action_model(valid_latent_action, actions_target)
            else:
                action_loss = 0.0 * latent_action.mean()

            total_loss = (
                action_loss.float()
                + transition_alignment_w * transition_alignment_loss.float()
                + transition_decode_w * transition_decode_loss.float()
            )

        return {
            "action_loss": action_loss,
            "transition_alignment_loss": transition_alignment_loss,
            "transition_alignment_cosine_loss": transition_alignment_cosine_loss,
            "transition_alignment_l1_loss": transition_alignment_l1_loss,
            "transition_decode_loss": transition_decode_loss,
            "transition_decode_l1_loss": transition_decode_l1_loss,
            "transition_decode_cosine_loss": transition_decode_cosine_loss,
            "transition_depth_decode_loss": transition_depth_decode_loss,
            "transition_depth_decode_pixel_loss": transition_depth_decode_pixel_loss,
            "transition_depth_decode_grad_loss": transition_depth_decode_grad_loss,
            "total_loss": total_loss,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> dict:
        """
        Inference forward: predict future actions from current multi-view images and instruction.
        Inference does not use DINO/Depth/LAM targets.
        """
        if type(examples) is not list:
            examples = [examples]

        B = len(examples)
        if B > 1:
            valid_indices = []
            valid_examples = []
            valid_actions = []

            for i, example in enumerate(examples):
                act = example.get("action", None)
                if act is not None:
                    valid_indices.append(i)
                    valid_examples.append(example)
                    valid_actions.append(act)

            if len(valid_indices) == 0:
                return {"normalized_actions": np.array([])}

            examples = valid_examples

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        states = [example.get("state", None) for example in examples]
        state_texts = [self._format_state_as_text(state) for state in states]

        def safe_get(imgs, idx):
            try:
                return imgs[idx][0][-1]
            except Exception:
                return None

        processed_batch_images = []
        for images in batch_images:
            head = safe_get(images, 0)
            left = safe_get(images, 1)
            right = safe_get(images, 2)

            ref_img = head or left or right
            if ref_img is not None:
                blank_img = Image.new(ref_img.mode, ref_img.size, color=0)
            else:
                blank_img = Image.new("RGB", (224, 224), color=0)

            processed_batch_images.append([
                head or blank_img,
                left or blank_img,
                right or blank_img,
            ])

        batch_images = processed_batch_images

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [
            instruction + state_text + prompt_suffix
            for instruction, state_text in zip(instructions, state_texts)
        ]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            input_ids = qwen_inputs.get("input_ids", None)
            latent_action = self._gather_token_embeddings(
                last_hidden, input_ids, self.action_token_id, self.chunk_len
            )


        with torch.autocast("cuda", dtype=torch.float32):
            latent_action = latent_action.float()
    
            pred_actions = self.action_model.predict_action(latent_action)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _gather_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
        expected_len: int,
    ) -> torch.Tensor:
        """Extract the hidden states at special token positions."""
        device = input_ids.device
        B, L, H = last_hidden.shape
        mask = input_ids == token_id

        counts = mask.sum(dim=1)
        if (counts < expected_len).any():
            raise RuntimeError(f"Token ID {token_id} appears fewer than {expected_len} times.")

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=expected_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values

        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        return last_hidden.gather(dim=1, index=expanded_index)


if __name__ == "__main__":
    pass
