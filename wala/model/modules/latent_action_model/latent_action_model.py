import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class ContinuousTransitionBottleneckTokenizer(nn.Module):
    """Continuous latent action tokenizer for semantic and geometric dynamics.

    The RGB branch encodes current DINOv3 patch features together with future
    feature deltas, then decodes future DINOv3 patch deltas from the latent
    tokens. When depth is enabled, depth maps are normalized per clip, converted
    into local geometry channels, patchified, and fused with the RGB latent
    tokens. The depth decoder predicts dense normalized depth residuals.

    Main tensor shapes:
      current_feats: (B, L, D)
      future_feats: (B, T, L, D)
      current_depth: (B, H, W), optional
      future_depth: (B, T, H, W), optional
      transition_tokens: (B, K, H)
      pred_delta: (B, T, L, D)
      pred_depth_delta: (B, T, H, W), optional
    """

    def __init__(
        self,
        *,
        dino_dim: int = 768,
        hidden_dim: int = 768,
        num_transition_tokens: int = 50,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        num_heads: int = 8,
        max_future_steps: int = 64,
        max_patches: int = 1024,
        dropout: float = 0.0,
        smooth_l1_weight: float = 1.0,
        cosine_weight: float = 0.1,
        composition_weight: float = 0.0,
        reencode_weight: float = 0.0,
        content_invariance_weight: float = 0.0,
        content_aug_scale_std: float = 0.05,
        content_aug_bias_std: float = 0.02,
        min_composition_steps: int = 2,
        use_depth: bool = False,
        depth_loss_weight: float = 1.0,
        depth_gradient_loss_weight: float = 0.2,
        depth_fusion_scale: float = 1.0,
        depth_patch_size: int = 14,
    ) -> None:
        super().__init__()

        self.dino_dim = dino_dim
        self.hidden_dim = hidden_dim
        self.num_transition_tokens = num_transition_tokens
        self.max_future_steps = max_future_steps
        self.max_patches = max_patches

        self.smooth_l1_weight = smooth_l1_weight
        self.cosine_weight = cosine_weight
        self.composition_weight = composition_weight
        self.reencode_weight = reencode_weight
        self.content_invariance_weight = content_invariance_weight
        self.content_aug_scale_std = content_aug_scale_std
        self.content_aug_bias_std = content_aug_bias_std
        self.min_composition_steps = min_composition_steps

        self.use_depth = use_depth
        self.depth_loss_weight = depth_loss_weight
        self.depth_gradient_loss_weight = depth_gradient_loss_weight
        self.depth_fusion_scale = depth_fusion_scale
        self.depth_patch_size = depth_patch_size if isinstance(depth_patch_size, tuple) else (depth_patch_size, depth_patch_size)

        # ---------------- RGB / DINO transition encoder ----------------
        self.current_input_proj = nn.Linear(dino_dim, hidden_dim)
        self.delta_input_proj = nn.Linear(dino_dim, hidden_dim)

        self.transition_queries = nn.Parameter(torch.randn(1, num_transition_tokens, hidden_dim) * 0.02)
        self.transition_token_embed = nn.Parameter(torch.randn(1, num_transition_tokens, hidden_dim) * 0.02)

        self.time_embed = nn.Parameter(torch.randn(1, max_future_steps, hidden_dim) * 0.02)
        self.patch_embed = nn.Parameter(torch.randn(1, max_patches, hidden_dim) * 0.02)
        self.current_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.delta_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.horizon_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerDecoder(encoder_layer, num_layers=encoder_layers)

        # ---------------- RGB / DINO transition decoder ----------------
        horizon_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.horizon_decoder = nn.TransformerDecoder(horizon_layer, num_layers=decoder_layers)

        self.current_patch_proj = nn.Linear(dino_dim, hidden_dim)
        self.transition_film = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, dino_dim),
        )

        # ---------------- Depth branch ----------------
        if self.use_depth:
            depth_patch_dim = 3 * self.depth_patch_size[0] * self.depth_patch_size[1]
            self.depth_current_patch_proj = nn.Sequential(
                nn.Linear(depth_patch_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.depth_delta_patch_proj = nn.Sequential(
                nn.Linear(depth_patch_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.depth_current_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.depth_delta_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))

            self.depth_transition_queries = nn.Parameter(
                torch.randn(1, num_transition_tokens, hidden_dim) * 0.02
            )
            self.depth_transition_token_embed = nn.Parameter(
                torch.randn(1, num_transition_tokens, hidden_dim) * 0.02
            )

            depth_encoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.depth_encoder = nn.TransformerDecoder(depth_encoder_layer, num_layers=encoder_layers)

            # Fuse RGB and depth transition tokens into one canonical transition token.
            self.rgb_depth_cross_fuse = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=1,
            )

            self.rgb_depth_fuse = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

            depth_spatial_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.depth_spatial_decoder = nn.TransformerDecoder(depth_spatial_layer, num_layers=1)

            self.depth_decode_patch_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.depth_patch_film = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
            )
            self.depth_patch_pixel_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 2),
                nn.Linear(hidden_dim + 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.depth_dense_refine = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 1, kernel_size=3, padding=1),
            )

    def _check_shapes(self, current_feats: torch.Tensor, future_feats: torch.Tensor) -> None:
        if current_feats.ndim != 3:
            raise ValueError(f"current_feats should be (B, L, D), got {tuple(current_feats.shape)}.")
        if future_feats.ndim != 4:
            raise ValueError(f"future_feats should be (B, T, L, D), got {tuple(future_feats.shape)}.")
        if current_feats.shape[0] != future_feats.shape[0]:
            raise ValueError("Batch size mismatch between current_feats and future_feats.")
        if current_feats.shape[1] != future_feats.shape[2]:
            raise ValueError("Patch length mismatch between current_feats and future_feats.")
        if current_feats.shape[2] != future_feats.shape[3]:
            raise ValueError("Feature dim mismatch between current_feats and future_feats.")

    def _future_delta(self, current_feats: torch.Tensor, future_feats: torch.Tensor) -> torch.Tensor:
        self._check_shapes(current_feats, future_feats)
        return future_feats - current_feats[:, None, :, :]

    def _time_pos(self, T: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if T > self.max_future_steps:
            raise ValueError(f"Future sequence length T={T} exceeds max_future_steps={self.max_future_steps}.")
        return self.time_embed[:, :T].to(device=device, dtype=dtype)

    def _patch_pos(self, L: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if L > self.max_patches:
            raise ValueError(f"Number of DINO patches L={L} exceeds max_patches={self.max_patches}.")
        return self.patch_embed[:, :L].to(device=device, dtype=dtype)

    def _infer_square_patch_hw(self, L: int) -> Tuple[int, int]:
        h = int(L ** 0.5)
        if h * h != L:
            raise ValueError(
                f"Cannot infer square patch grid from L={L}. Pass depth_patch_hw=(patch_h, patch_w)."
            )
        return h, h

    def _normalize_depth_clip(
        self,
        current_depth: torch.Tensor,
        future_depth: torch.Tensor,
        eps: float = 1e-6,
        clamp: float = 3.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize depth per clip, not per frame, so current-to-future residual remains meaningful.

        current_depth: [B,H,W]
        future_depth:  [B,T,H,W]
        """
        if current_depth.ndim != 3:
            raise ValueError(f"current_depth should be [B,H,W], got {tuple(current_depth.shape)}.")
        if future_depth.ndim != 4:
            raise ValueError(f"future_depth should be [B,T,H,W], got {tuple(future_depth.shape)}.")
        if current_depth.shape[0] != future_depth.shape[0]:
            raise ValueError("Batch size mismatch between current_depth and future_depth.")

        depth = torch.cat([current_depth[:, None], future_depth], dim=1).float()
        depth = torch.clamp(depth, min=eps)
        log_d = torch.log(depth)

        flat = log_d.flatten(1)
        median_1d = flat.median(dim=1).values
        median = median_1d[:, None, None, None]
        mad = (flat - median_1d[:, None]).abs().median(dim=1).values[:, None, None, None]

        norm_d = (log_d - median) / (mad + eps)
        norm_d = torch.clamp(norm_d, -clamp, clamp)
        return norm_d[:, 0], norm_d[:, 1:]

    def _depth_geometry_input(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Build local geometry channels from a normalized depth map.

        depth: [B,H,W]
        return: [B,3,H,W] with depth, dx, dy
        """
        if depth.ndim != 3:
            raise ValueError(f"depth should be [B,H,W], got {tuple(depth.shape)}.")

        B, H, W = depth.shape
        depth_4d = depth[:, None]
        if W > 1:
            dx = depth_4d[:, :, :, 1:] - depth_4d[:, :, :, :-1]
            dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
        else:
            dx = torch.zeros_like(depth_4d)
        if H > 1:
            dy = depth_4d[:, :, 1:, :] - depth_4d[:, :, :-1, :]
            dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
        else:
            dy = torch.zeros_like(depth_4d)
        return torch.cat([depth_4d, dx, dy], dim=1)

    def _depth_patch_size(self, depth_hw: Tuple[int, int], patch_hw: Tuple[int, int]) -> Tuple[int, int]:
        H, W = depth_hw
        patch_h, patch_w = patch_hw
        if H % patch_h != 0 or W % patch_w != 0:
            raise ValueError(
                f"Depth resolution {(H, W)} must be divisible by depth_patch_hw={patch_hw} for ViT-style patchify."
            )
        patch_size = H // patch_h, W // patch_w
        if patch_size != self.depth_patch_size:
            raise ValueError(
                f"Depth patch pixel size {patch_size} does not match configured "
                f"depth_patch_size={self.depth_patch_size}."
            )
        return patch_size

    def _depth_map_to_patch_tokens(
        self,
        depth: torch.Tensor,
        *,
        patch_hw: Tuple[int, int],
        projector: nn.Module,
    ) -> torch.Tensor:
        """
        depth: [B,H,W] or [B,T,H,W]
        return: [B,L,H] or [B,T,L,H]
        """
        projector_param = next(projector.parameters())
        depth = depth.to(device=projector_param.device, dtype=projector_param.dtype)

        if depth.ndim == 3:
            _, H, W = depth.shape
            patch_size = self._depth_patch_size((H, W), patch_hw)
            patches = F.unfold(self._depth_geometry_input(depth), kernel_size=patch_size, stride=patch_size)
            patches = patches.transpose(1, 2)
            return projector(patches)

        if depth.ndim == 4:
            B, T, H, W = depth.shape
            depth_3d = depth.reshape(B * T, H, W)
            patch_size = self._depth_patch_size((H, W), patch_hw)
            patches = F.unfold(self._depth_geometry_input(depth_3d), kernel_size=patch_size, stride=patch_size)
            patches = patches.transpose(1, 2)
            tokens = projector(patches)
            return tokens.view(B, T, tokens.shape[1], tokens.shape[2])

        raise ValueError(f"depth should be [B,H,W] or [B,T,H,W], got {tuple(depth.shape)}.")

    def _prepare_depth_features(
        self,
        current_depth: torch.Tensor,
        future_depth: torch.Tensor,
        *,
        L: int,
        depth_patch_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if depth_patch_hw is None:
            depth_patch_hw = self._infer_square_patch_hw(L)
        if depth_patch_hw[0] * depth_patch_hw[1] != L:
            raise ValueError(f"depth_patch_hw={depth_patch_hw} is incompatible with DINO patch length L={L}.")

        current_depth, future_depth = self._normalize_depth_clip(current_depth, future_depth)
        depth_delta = future_depth - current_depth[:, None, :, :]
        current_depth_tokens = self._depth_map_to_patch_tokens(
            current_depth,
            patch_hw=depth_patch_hw,
            projector=self.depth_current_patch_proj,
        )
        depth_delta_tokens = self._depth_map_to_patch_tokens(
            depth_delta,
            patch_hw=depth_patch_hw,
            projector=self.depth_delta_patch_proj,
        )
        return current_depth_tokens, depth_delta_tokens

    def _shared_content_augment(
        self,
        current_feats: torch.Tensor,
        future_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.content_aug_scale_std <= 0 and self.content_aug_bias_std <= 0:
            return current_feats, future_feats

        B, _, D = current_feats.shape
        dtype, device = current_feats.dtype, current_feats.device
        scale = 1.0 + self.content_aug_scale_std * torch.randn(B, 1, D, device=device, dtype=dtype)
        bias = self.content_aug_bias_std * torch.randn(B, 1, D, device=device, dtype=dtype)
        cur_aug = current_feats * scale + bias
        fut_aug = future_feats * scale[:, None, :, :] + bias[:, None, :, :]
        return cur_aug, fut_aug

    def _token_cosine_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = F.normalize(pred.float(), dim=-1)
        target_norm = F.normalize(target.float(), dim=-1)
        return 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()

    def _depth_gradient_l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

    def _encode_rgb(self, current_feats: torch.Tensor, future_feats: torch.Tensor) -> torch.Tensor:
        delta = self._future_delta(current_feats, future_feats)
        B, T, L, D = delta.shape
        dtype = current_feats.dtype
        device = current_feats.device

        time_pos = self._time_pos(T, device=device, dtype=dtype)
        patch_pos = self._patch_pos(L, device=device, dtype=dtype)

        current_memory = self.current_input_proj(current_feats)
        current_memory = current_memory + patch_pos + self.current_type_embed.to(device=device, dtype=dtype)

        flat_delta = delta.reshape(B, T * L, D)
        delta_memory = self.delta_input_proj(flat_delta)
        delta_pos = (time_pos[:, :, None, :] + patch_pos[:, None, :, :]).reshape(1, T * L, self.hidden_dim)
        delta_memory = delta_memory + delta_pos + self.delta_type_embed.to(device=device, dtype=dtype)

        memory = torch.cat([current_memory, delta_memory], dim=1)

        queries = self.transition_queries.to(device=device, dtype=dtype).expand(B, -1, -1)
        queries = queries + self.transition_token_embed.to(device=device, dtype=dtype).expand(B, -1, -1)
        return self.encoder(tgt=queries, memory=memory)

    def _encode_depth_tokens(
        self,
        current_depth_tokens: torch.Tensor,
        depth_delta_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        current_depth_tokens: [B,L,H]
        depth_delta_tokens:   [B,T,L,H]
        """
        B, T, L, _ = depth_delta_tokens.shape
        device = current_depth_tokens.device
        dtype = current_depth_tokens.dtype

        time_pos = self._time_pos(T, device=device, dtype=dtype)
        patch_pos = self._patch_pos(L, device=device, dtype=dtype)

        current_memory = current_depth_tokens + patch_pos + self.depth_current_type_embed.to(device=device, dtype=dtype)

        delta_memory = depth_delta_tokens.reshape(B, T * L, self.hidden_dim)
        delta_pos = (time_pos[:, :, None, :] + patch_pos[:, None, :, :]).reshape(1, T * L, self.hidden_dim)
        delta_memory = delta_memory + delta_pos + self.depth_delta_type_embed.to(device=device, dtype=dtype)

        memory = torch.cat([current_memory, delta_memory], dim=1)
        queries = self.depth_transition_queries.to(device=device, dtype=dtype).expand(B, -1, -1)
        queries = queries + self.depth_transition_token_embed.to(device=device, dtype=dtype).expand(B, -1, -1)
        return self.depth_encoder(tgt=queries, memory=memory)

    def encode(
        self,
        current_feats: torch.Tensor,
        future_feats: torch.Tensor,
        current_depth: Optional[torch.Tensor] = None,
        future_depth: Optional[torch.Tensor] = None,
        depth_patch_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        E(F_t, F_future - F_t, optional D_t, D_future-D_t) -> compact transition tokens.
        """
        rgb_transition_tokens = self._encode_rgb(current_feats, future_feats)

        if not self.use_depth or current_depth is None or future_depth is None:
            return rgb_transition_tokens

        _, _, L, _ = future_feats.shape
        device = current_feats.device
        dtype = current_feats.dtype
        current_depth = current_depth.to(device=device)
        future_depth = future_depth.to(device=device)

        current_depth_tokens, depth_delta_tokens = self._prepare_depth_features(
            current_depth=current_depth,
            future_depth=future_depth,
            L=L,
            depth_patch_hw=depth_patch_hw,
        )
        current_depth_tokens = current_depth_tokens.to(device=device, dtype=dtype)
        depth_delta_tokens = depth_delta_tokens.to(device=device, dtype=dtype)

        depth_transition_tokens = self._encode_depth_tokens(current_depth_tokens, depth_delta_tokens)

        depth_context = self.rgb_depth_cross_fuse(
            tgt=rgb_transition_tokens,
            memory=depth_transition_tokens,
        )

        fused_residual = self.rgb_depth_fuse(
            torch.cat([rgb_transition_tokens, depth_context], dim=-1)
        )

        return rgb_transition_tokens + self.depth_fusion_scale * fused_residual

    def _depth_spatial_tokens(
        self,
        transition_tokens: torch.Tensor,
        *,
        num_future_steps: int,
        patch_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Generate per-future-step patch tokens from transition tokens.

        return: [B,T,L,H]
        """
        B = transition_tokens.shape[0]
        patch_h, patch_w = patch_hw
        L = patch_h * patch_w
        device = transition_tokens.device
        dtype = transition_tokens.dtype

        time_pos = self._time_pos(num_future_steps, device=device, dtype=dtype)
        patch_pos = self._patch_pos(L, device=device, dtype=dtype)
        spatial_queries = (time_pos[:, :, None, :] + patch_pos[:, None, :, :]).reshape(
            1, num_future_steps * L, self.hidden_dim
        )
        spatial_queries = spatial_queries.expand(B, -1, -1)

        spatial_tokens = self.depth_spatial_decoder(tgt=spatial_queries, memory=transition_tokens)
        return spatial_tokens.view(B, num_future_steps, L, self.hidden_dim)

    def decode_delta(
        self,
        current_feats: torch.Tensor,
        transition_tokens: torch.Tensor,
        *,
        num_future_steps: int,
    ) -> torch.Tensor:
        """
        D_rgb(F_t, z, h) -> future DINO patch deltas.
        """
        B, L, D = current_feats.shape
        device = current_feats.device
        dtype = current_feats.dtype

        time_pos = self._time_pos(num_future_steps, device=device, dtype=dtype)
        patch_pos = self._patch_pos(L, device=device, dtype=dtype)

        horizon_queries = self.horizon_query.to(device=device, dtype=dtype) + time_pos
        horizon_queries = horizon_queries.expand(B, -1, -1)
        horizon_context = self.horizon_decoder(tgt=horizon_queries, memory=transition_tokens)

        patch_hidden = self.current_patch_proj(current_feats) + patch_pos
        gamma_beta = self.transition_film(horizon_context)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        hidden = patch_hidden[:, None, :, :] * (1.0 + gamma[:, :, None, :]) + beta[:, :, None, :]
        return self.delta_head(hidden)

    def _depth_patch_tokens_to_dense(
        self,
        patch_tokens: torch.Tensor,
        *,
        patch_hw: Tuple[int, int],
        depth_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Decode patch tokens to dense depth residuals with overlapping tile blending.

        patch_tokens: [B,T,L,H]
        return: [B,T,H,W]
        """
        B, T, L, _ = patch_tokens.shape
        H, W = depth_hw
        patch_h, patch_w = patch_hw
        stride_size = self._depth_patch_size(depth_hw, patch_hw)
        pad_size = (stride_size[0] // 4, stride_size[1] // 4)
        kernel_size = (
            stride_size[0] + 2 * pad_size[0],
            stride_size[1] + 2 * pad_size[1],
        )
        patch_pixels = kernel_size[0] * kernel_size[1]
        if L != patch_h * patch_w:
            raise ValueError(f"patch token length L={L} is incompatible with depth_patch_hw={patch_hw}.")

        device = patch_tokens.device
        dtype = patch_tokens.dtype
        y = torch.linspace(-1.0, 1.0, kernel_size[0], device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, kernel_size[1], device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, 1, 1, patch_pixels, 2)

        token_pixels = patch_tokens[:, :, :, None, :].expand(-1, -1, -1, patch_pixels, -1)
        coords = coords.expand(B, T, L, -1, -1)
        pixel_input = torch.cat([token_pixels, coords], dim=-1)
        patch_values = self.depth_patch_pixel_head(pixel_input).squeeze(-1)

        patch_values = patch_values.reshape(B * T, L, patch_pixels).transpose(1, 2)
        dense_sum = F.fold(
            patch_values,
            output_size=(H, W),
            kernel_size=kernel_size,
            padding=pad_size,
            stride=stride_size,
        )

        ones = torch.ones(
            B * T,
            patch_pixels,
            L,
            device=device,
            dtype=dtype,
        )
        dense_weight = F.fold(
            ones,
            output_size=(H, W),
            kernel_size=kernel_size,
            padding=pad_size,
            stride=stride_size,
        ).clamp_min(1.0)
        dense = dense_sum / dense_weight
        return dense[:, 0].view(B, T, H, W)

    def _decode_dense_depth_delta(
        self,
        current_depth: torch.Tensor,
        transition_tokens: torch.Tensor,
        *,
        num_future_steps: int,
        patch_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        D_dep(D_t, z, h) -> dense normalized depth residual.

        current_depth: [B,H,W]
        return: [B,T,H,W]
        """
        if not self.use_depth:
            raise RuntimeError("_decode_dense_depth_delta() called while use_depth=False.")
        if current_depth.ndim != 3:
            raise ValueError(f"current_depth should be [B,H,W], got {tuple(current_depth.shape)}.")

        B, H, W = current_depth.shape
        device = transition_tokens.device
        dtype = transition_tokens.dtype
        current_depth = current_depth.to(device=device, dtype=dtype)

        current_depth_tokens = self._depth_map_to_patch_tokens(
            current_depth,
            patch_hw=patch_hw,
            projector=self.depth_current_patch_proj,
        )
        current_depth_tokens = current_depth_tokens[:, None, :, :].expand(-1, num_future_steps, -1, -1)
        motion_tokens = self._depth_spatial_tokens(
            transition_tokens,
            num_future_steps=num_future_steps,
            patch_hw=patch_hw,
        )
        gamma_beta = self.depth_patch_film(motion_tokens)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        patch_tokens = current_depth_tokens * (1.0 + gamma) + beta
        patch_tokens = self.depth_decode_patch_proj(patch_tokens)
        coarse_delta = self._depth_patch_tokens_to_dense(patch_tokens, patch_hw=patch_hw, depth_hw=(H, W))

        current_depth_bt = current_depth[:, None, :, :].expand(-1, num_future_steps, -1, -1)
        coarse_future = current_depth_bt + coarse_delta
        refine_input = torch.stack(
            [current_depth_bt, coarse_delta, coarse_future],
            dim=2,
        ).reshape(B * num_future_steps, 3, H, W)
        refine_delta = self.depth_dense_refine(refine_input).view(B, num_future_steps, H, W)
        return coarse_delta + refine_delta

    def _composition_loss(
        self,
        current_feats: torch.Tensor,
        future_feats: torch.Tensor,
        full_pred_delta: torch.Tensor,
    ) -> torch.Tensor:
        B, T, L, D = future_feats.shape
        if T < self.min_composition_steps:
            return full_pred_delta.new_tensor(0.0)

        split = max(1, T // 2)
        if split >= T:
            return full_pred_delta.new_tensor(0.0)

        prefix_future = future_feats[:, :split, :, :]
        z_prefix = self.encode(current_feats, prefix_future)
        pred_prefix_delta = self.decode_delta(current_feats, z_prefix, num_future_steps=split)
        pred_mid_feats = current_feats + pred_prefix_delta[:, -1, :, :]

        suffix_current_gt = future_feats[:, split - 1, :, :]
        suffix_future_gt = future_feats[:, split:, :, :]
        z_suffix = self.encode(suffix_current_gt.detach(), suffix_future_gt.detach())
        pred_suffix_delta = self.decode_delta(pred_mid_feats, z_suffix, num_future_steps=T - split)
        sequential_final_feats = pred_mid_feats + pred_suffix_delta[:, -1, :, :]

        direct_final_feats = current_feats + full_pred_delta[:, -1, :, :]
        return F.smooth_l1_loss(direct_final_feats.float(), sequential_final_feats.float())

    def _reencode_loss(
        self,
        current_feats: torch.Tensor,
        pred_delta: torch.Tensor,
        transition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        pred_future_feats = current_feats[:, None, :, :] + pred_delta
        reencoded_tokens = self.encode(current_feats.detach(), pred_future_feats)
        return self._token_cosine_loss(reencoded_tokens, transition_tokens.detach()) + 0.25 * F.smooth_l1_loss(
            reencoded_tokens.float(), transition_tokens.float()
        )

    def _content_invariance_loss(
        self,
        current_feats: torch.Tensor,
        future_feats: torch.Tensor,
        transition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        current_aug, future_aug = self._shared_content_augment(current_feats, future_feats)
        aug_tokens = self.encode(current_aug, future_aug)
        return self._token_cosine_loss(aug_tokens, transition_tokens.detach()) + 0.25 * F.smooth_l1_loss(
            aug_tokens.float(), transition_tokens.float()
        )

    def forward(
        self,
        current_feats: torch.Tensor,
        future_feats: torch.Tensor,
        current_depth: Optional[torch.Tensor] = None,
        future_depth: Optional[torch.Tensor] = None,
        depth_patch_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pretrain the continuous transition bottleneck."""
        delta = self._future_delta(current_feats, future_feats).detach()

        depth_active = self.use_depth and current_depth is not None and future_depth is not None

        transition_tokens = self.encode(
            current_feats,
            future_feats,
            current_depth=current_depth if depth_active else None,
            future_depth=future_depth if depth_active else None,
            depth_patch_hw=depth_patch_hw,
        )
        pred_delta = self.decode_delta(current_feats, transition_tokens, num_future_steps=future_feats.shape[1])

        recon_loss = F.l1_loss(pred_delta.float(), delta.float())
        cosine_loss = self._token_cosine_loss(pred_delta, delta)

        aux_can_run = not depth_active

        composition_loss = pred_delta.new_tensor(0.0)
        if self.composition_weight > 0 and aux_can_run:
            composition_loss = self._composition_loss(current_feats, future_feats, pred_delta)

        reencode_loss = pred_delta.new_tensor(0.0)
        if self.reencode_weight > 0 and aux_can_run:
            reencode_loss = self._reencode_loss(current_feats, pred_delta, transition_tokens)

        content_invariance_loss = pred_delta.new_tensor(0.0)
        if self.content_invariance_weight > 0 and aux_can_run:
            content_invariance_loss = self._content_invariance_loss(current_feats, future_feats, transition_tokens)

        depth_recon_loss = pred_delta.new_tensor(0.0)
        depth_pixel_loss = pred_delta.new_tensor(0.0)
        depth_grad_loss = pred_delta.new_tensor(0.0)
        pred_depth_delta = None
        target_depth_delta = None

        if depth_active:
            B, T, L, _ = future_feats.shape
            device = current_feats.device
            if depth_patch_hw is None:
                depth_patch_hw = self._infer_square_patch_hw(L)
            if depth_patch_hw[0] * depth_patch_hw[1] != L:
                raise ValueError(f"depth_patch_hw={depth_patch_hw} is incompatible with DINO patch length L={L}.")

            current_depth_norm, future_depth_norm = self._normalize_depth_clip(
                current_depth=current_depth.to(device=device),
                future_depth=future_depth.to(device=device),
            )
            target_depth_delta = (future_depth_norm - current_depth_norm[:, None, :, :]).detach()
            pred_depth_delta = self._decode_dense_depth_delta(
                current_depth=current_depth_norm,
                transition_tokens=transition_tokens,
                num_future_steps=T,
                patch_hw=depth_patch_hw,
            )
            target_depth_delta = target_depth_delta.to(device=device, dtype=pred_depth_delta.dtype)
            depth_pixel_loss = F.smooth_l1_loss(pred_depth_delta.float(), target_depth_delta.float())
            depth_grad_loss = self._depth_gradient_l1_loss(pred_depth_delta, target_depth_delta)
            depth_recon_loss = depth_pixel_loss + self.depth_gradient_loss_weight * depth_grad_loss

        loss = (
            self.smooth_l1_weight * recon_loss
            + self.cosine_weight * cosine_loss
            + self.composition_weight * composition_loss
            + self.reencode_weight * reencode_loss
            + self.content_invariance_weight * content_invariance_loss
            + self.depth_loss_weight * depth_recon_loss
        )

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "cosine_loss": cosine_loss,
            "composition_loss": composition_loss,
            "reencode_loss": reencode_loss,
            "content_invariance_loss": content_invariance_loss,
            "depth_recon_loss": depth_recon_loss,
            "depth_pixel_loss": depth_pixel_loss,
            "depth_grad_loss": depth_grad_loss,
            "transition_tokens": transition_tokens,
            "pred_delta": pred_delta,
            "pred_depth_delta": pred_depth_delta,
            "target_depth_delta": target_depth_delta,
        }


class LatentActionToTransitionTokens(nn.Module):
    """Resample VLA latent action tokens into the transition-token space."""

    def __init__(
        self,
        *,
        action_dim: int,
        transition_dim: int,
        num_transition_tokens: int,
        num_layers: int = 1,
        num_heads: int = 8,
        max_action_tokens: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_transition_tokens = num_transition_tokens
        self.max_action_tokens = max_action_tokens

        self.action_proj = nn.Linear(action_dim, transition_dim)
        self.action_time_embed = nn.Parameter(torch.randn(1, max_action_tokens, transition_dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, num_transition_tokens, transition_dim) * 0.02)
        self.query_embed = nn.Parameter(torch.randn(1, num_transition_tokens, transition_dim) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=transition_dim,
            nhead=num_heads,
            dim_feedforward=transition_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.resampler = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, latent_action: torch.Tensor) -> torch.Tensor:
        if latent_action.ndim != 3:
            raise ValueError(f"latent_action should be (B, T, H), got {tuple(latent_action.shape)}.")
        B, T, _ = latent_action.shape
        if T > self.max_action_tokens:
            raise ValueError(f"Action chunk length T={T} exceeds max_action_tokens={self.max_action_tokens}.")

        memory = self.action_proj(latent_action)
        memory = memory + self.action_time_embed[:, :T].to(device=memory.device, dtype=memory.dtype)

        query = self.query.to(device=memory.device, dtype=memory.dtype).expand(B, -1, -1)
        query = query + self.query_embed.to(device=memory.device, dtype=memory.dtype).expand(B, -1, -1)
        return self.resampler(tgt=query, memory=memory)
