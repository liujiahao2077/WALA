# Copyright 2026 WALA authors. All rights reserved.
# Licensed under the MIT License.

from typing import Optional

import torch
from wala.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Reserved token range for discretized action tokens in the Qwen3-VL tokenizer.
_ACTION_TOKEN_MIN = 151669
_ACTION_TOKEN_MAX = 153716


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

                # Add learnable text-side special tokens for missing camera views.
        # When an all-zero image is detected, the image will NOT be fed into VLM.
        # Instead, the view-specific token will be inserted at the corresponding position.
        self.num_missing_view_tokens = int(qwenvl_config.get("num_missing_view_tokens", 3))

        self.missing_view_tokens = [
            f"<|missing_view_{i}|>" for i in range(self.num_missing_view_tokens)
        ]

        old_vocab_size = len(self.processor.tokenizer)

        existing_additional_special_tokens = list(
            self.processor.tokenizer.additional_special_tokens
        )

        for token in self.missing_view_tokens:
            if token not in existing_additional_special_tokens:
                existing_additional_special_tokens.append(token)

        num_added_tokens = self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": existing_additional_special_tokens}
        )

        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(self.processor.tokenizer))

        self.missing_view_token_ids = [
            self.processor.tokenizer.convert_tokens_to_ids(token)
            for token in self.missing_view_tokens
        ]

        # Initialize each new missing-view token embedding with the average embedding
        # of a semantic phrase such as "missing view 0", instead of random initialization.
        if num_added_tokens > 0:
            with torch.no_grad():
                input_embeddings = self.model.get_input_embeddings()

                for view_idx, token_id in enumerate(self.missing_view_token_ids):
                    # Only initialize newly added token rows.
                    if token_id < old_vocab_size:
                        continue

                    init_ids = self.processor.tokenizer(
                        f"missing view {view_idx}",
                        add_special_tokens=False,
                        return_tensors=None,
                    )["input_ids"]

                    if len(init_ids) > 0:
                        init_ids = torch.tensor(
                            init_ids,
                            dtype=torch.long,
                            device=input_embeddings.weight.device,
                        )
                        input_embeddings.weight[token_id].copy_(
                            input_embeddings.weight[init_ids].mean(dim=0)
                        )
        
        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    @staticmethod
    def _is_all_zero_image(img, eps: float = 1e-6) -> bool:
        """
        Detect whether an input image is an all-zero placeholder.

        Optimized version:
            - PIL.Image: use getbbox(), avoid np.asarray + float32 copy.
            - numpy integer array: use count_nonzero(), avoid astype/abs/nan_to_num.
            - torch.Tensor: use count_nonzero() for exact zero placeholder.
        """
        if isinstance(img, torch.Tensor):
            img = img.detach()
            return bool(torch.count_nonzero(img).item() == 0)

        try:
            from PIL import Image

            if isinstance(img, Image.Image):
                if img.mode == "RGBA":
                    return (
                        img.getchannel("R").getbbox() is None
                        and img.getchannel("G").getbbox() is None
                        and img.getchannel("B").getbbox() is None
                    )

                return img.getbbox() is None

        except Exception:
            pass

        import numpy as np

        arr = np.asarray(img)
        if arr.size == 0:
            return False

        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]

        if np.issubdtype(arr.dtype, np.integer):
            return bool(np.count_nonzero(arr) == 0)

        return bool(np.nanmax(np.abs(arr)) <= eps)
    
    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = []
            for view_idx, img in enumerate(imgs):
                if self._is_all_zero_image(img):
                    if view_idx >= len(self.missing_view_tokens):
                        raise ValueError(
                            f"view_idx={view_idx} exceeds the number of missing-view tokens "
                            f"{len(self.missing_view_tokens)}. "
                            f"Please increase framework.qwenvl.num_missing_view_tokens."
                        )

                    missing_view_token = self.missing_view_tokens[view_idx]
                    content.append(
                        {
                            "type": "text",
                            "text": f"View {view_idx} is missing. {missing_view_token}\n",
                        }
                    )
                else:
                    content.append({"type": "image", "image": img})
            
            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:
            action_token_min = _ACTION_TOKEN_MIN
            action_token_max = _ACTION_TOKEN_MAX
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "No action tokens were found in the tokenizer output; the whole sequence is ignored for loss."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels
        
        return batch_inputs.to(self.model.device)
