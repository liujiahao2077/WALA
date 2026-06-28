from typing import Any

import numpy as np
from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """Recursively resize a PIL image, numpy image, or nested image list."""
    if isinstance(images, np.ndarray):
        images = Image.fromarray(images)
    if isinstance(images, Image.Image):
        return images.resize(target_size)
    if isinstance(images, list):
        return [resize_images(img, target_size) for img in images]
    raise ValueError("Unsupported image type or structure.")


def to_pil_preserve(images: Any, scale_float: bool = True):
    """Convert numpy image arrays to PIL without resizing or changing nesting layout."""
    if isinstance(images, Image.Image):
        return images
    if isinstance(images, np.ndarray):
        arr = images
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            if scale_float:
                arr = (arr * 255.0).round().astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
            return Image.fromarray(arr)
        raise ValueError(f"Unsupported image array shape: {arr.shape}")
    if isinstance(images, list):
        return [to_pil_preserve(x, scale_float=scale_float) for x in images]
    if isinstance(images, tuple):
        return tuple(to_pil_preserve(x, scale_float=scale_float) for x in images)
    raise TypeError(f"Unsupported image type: {type(images)!r}")
