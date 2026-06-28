def get_vlm_model(config):
    """Build the VLM backbone used by WALA.

    The released core model uses a Qwen3-VL action-tokenized backbone.
    Other backbones from the research workspace are intentionally
    omitted from this compact source release.
    """
    vlm_name = config.framework.qwenvl.base_vlm
    if "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)
    raise NotImplementedError(
        f"WALA release includes the Qwen3-VL backend only, got {vlm_name!r}."
    )
