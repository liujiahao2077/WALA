"""
DINOv3 vision backbone wrapper using Hugging Face Transformers.

The model and preprocessing configuration are loaded from the same Hugging
Face model directory. No local DINOv3 source-code checkout is required.
"""

from typing import List, Sequence, Union

import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel


ImageInput = Union[Image.Image, "torch.Tensor"]


class DINOv3BackBone(nn.Module):
    """
    Frozen DINOv3 ViT wrapper exposing patch-token features.

    Args:
        model_name_or_path:
            Hugging Face model id or a local directory containing config.json,
            model weights, and preprocessor_config.json.
        freeze:
            Freeze the encoder and always run it in evaluation mode.
        local_files_only:
            Do not access Hugging Face Hub. Use this for an already downloaded
            local model directory or populated Hugging Face cache.
        token:
            Optional Hugging Face access token. It can also be supplied through
            the HF_TOKEN environment variable.
    """

    def __init__(
        self,
        model_name_or_path: str = "/mnt/data/ljh/dinov3_model/dinov3-vitb16-pretrain-lvd1689m",
        freeze: bool = True,
        local_files_only: bool = True,
        token: str = None,
    ) -> None:
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.freeze = freeze

        load_kwargs = {
            "local_files_only": local_files_only,
        }
        if token is not None:
            load_kwargs["token"] = token

        # In DDP, let rank 0 populate the shared Hugging Face cache first.
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if is_dist and not local_files_only:
            if rank == 0:
                AutoImageProcessor.from_pretrained(model_name_or_path, **load_kwargs)
                AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
            dist.barrier()
            load_kwargs["local_files_only"] = True

        self.processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        self.body = AutoModel.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )

        config = self.body.config
        if not hasattr(config, "patch_size") or not hasattr(config, "hidden_size"):
            raise ValueError(
                "This wrapper expects a DINOv3 ViT model with patch_size and "
                f"hidden_size in its config, got {type(config).__name__}."
            )

        self.patch_size = config.patch_size
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0))
        self.num_channels = int(config.hidden_size)

        self.body.eval()
        if freeze:
            for parameter in self.body.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval mode when the parent model trains."""
        super().train(mode)
        if self.freeze:
            self.body.eval()
        return self

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Processor-produced tensor [N, 3, H, W].

        Returns:
            DINOv3 patch tokens [N, num_patches, hidden_size], excluding the
            CLS token and register tokens.
        """
        outputs = self.body(pixel_values=pixel_values, return_dict=True)
        hidden = outputs.last_hidden_state
        num_prefix_tokens = 1 + self.num_register_tokens
        patch_tokens = hidden[:, num_prefix_tokens:, :]

        patch_h, patch_w = self._patch_grid(pixel_values.shape[-2:])
        expected_tokens = patch_h * patch_w
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "Unexpected DINOv3 patch-token count: "
                f"got {patch_tokens.shape[1]}, expected {expected_tokens} "
                f"for input {tuple(pixel_values.shape[-2:])} and "
                f"patch_size={self.patch_size}."
            )

        return patch_tokens

    def _patch_grid(self, image_hw) -> tuple:
        if isinstance(self.patch_size, int):
            patch_h = patch_w = self.patch_size
        else:
            patch_h, patch_w = self.patch_size
        return image_hw[0] // patch_h, image_hw[1] // patch_w

    def prepare_dino_input(
        self,
        img_list: Sequence[Sequence[ImageInput]],
    ) -> torch.Tensor:
        """
        Preprocess nested image sequences with the model's built-in processor.

        Args:
            img_list:
                Batch of image sequences/views:
                [[image_0, image_1, ...], [...], ...].

        Returns:
            Flattened pixel tensor [B * num_images, 3, H, W].
        """
        if len(img_list) == 0:
            raise ValueError("img_list is empty.")

        num_images = len(img_list[0])
        if num_images == 0:
            raise ValueError("The first image sequence is empty.")
        if any(len(images) != num_images for images in img_list):
            raise ValueError("All samples must contain the same number of images.")

        flat_images: List[ImageInput] = [
            image
            for images in img_list
            for image in images
        ]

        processor_output = self.processor(
            images=flat_images,
            return_tensors="pt",
        )
        pixel_values = processor_output["pixel_values"]

        parameter = next(self.body.parameters())
        return pixel_values.to(
            device=parameter.device,
            dtype=parameter.dtype,
            non_blocking=True,
        )


def get_dino_model(
    backone_name: str = "/mnt/data/ljh/dinov3_model/dinov3-vitb16-pretrain-lvd1689m",
    *,
    local_files_only: bool = True,
    token: str = None,
) -> DINOv3BackBone:
    """
    Factory compatible with the existing DINOv2 call site.

    `backone_name` can be either a Hugging Face model id or a local model
    directory.
    """
    return DINOv3BackBone(
        model_name_or_path=backone_name,
        freeze=True,
        local_files_only=local_files_only,
        token=token,
    )
