from abc import ABC, abstractmethod

from wala.dataloader.gr00t_lerobot.transform.base import ModalityTransform


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


# Benchmark-specific robot configs are auto-discovered from
# examples/*/train_files/data_registry/.
ROBOT_TYPE_CONFIG_MAP = {}
