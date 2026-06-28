import importlib
import pkgutil
from pathlib import Path

from transformers import PretrainedConfig, PreTrainedModel

from WALA.model.tools import FRAMEWORK_REGISTRY


_FRAMEWORKS_IMPORTED = False


def _auto_import_framework_modules() -> None:
    global _FRAMEWORKS_IMPORTED
    if _FRAMEWORKS_IMPORTED:
        return

    framework_dir = Path(__file__).resolve().parent
    for _, module_name, is_pkg in pkgutil.iter_modules([str(framework_dir)]):
        if module_name.startswith("_") or module_name in {"base_framework", "share_tools"}:
            continue
        if is_pkg:
            sub_dir = framework_dir / module_name
            for _, sub_name, _ in pkgutil.iter_modules([str(sub_dir)]):
                if not sub_name.startswith("_"):
                    importlib.import_module(f"WALA.model.framework.{module_name}.{sub_name}")
        else:
            importlib.import_module(f"WALA.model.framework.{module_name}")

    _FRAMEWORKS_IMPORTED = True


def build_framework(cfg):
    if not hasattr(cfg, "framework") or not hasattr(cfg.framework, "name"):
        raise ValueError("Missing `cfg.framework.name`.")

    _auto_import_framework_modules()
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        available = sorted(FRAMEWORK_REGISTRY._registry.keys())
        raise NotImplementedError(
            f"Framework `{framework_id}` is not implemented. Available frameworks: {available}"
        )
    return FRAMEWORK_REGISTRY[framework_id](cfg)


class baseframework(PreTrainedModel):
    """Minimal base class for WALA model assemblies."""

    def __init__(self, hf_config=PretrainedConfig()) -> None:
        super().__init__(hf_config)
