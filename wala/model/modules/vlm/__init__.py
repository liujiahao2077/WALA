"""Vision-language backbone factory for WALA."""


def get_vlm_model(config):
    vlm_name = str(config.framework.qwenvl.base_vlm)
    if "Qwen3-VL" not in vlm_name:
        raise NotImplementedError(
            f"Unsupported vision-language backend {vlm_name!r}; expected a Qwen3-VL model."
        )
    from .qwen3_vl import _QWen3_VL_Interface

    return _QWen3_VL_Interface(config)
