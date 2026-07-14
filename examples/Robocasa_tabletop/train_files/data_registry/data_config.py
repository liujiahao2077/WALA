"""Fourier GR1 / Robocasa — data config, embodiment tags, and mixtures."""

from wala.dataloader.gr00t_lerobot.datasets import ModalityConfig
from wala.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from wala.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from wala.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class FourierGr1ArmsWaistDataConfig:
    video_keys = ["video.ego_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist", 
                  "state.wrist_l_pos", "state.wrist_l_rot6d", "state.wrist_r_pos", "state.wrist_r_rot6d"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand", "action.waist", 
                   "action.wrist_l_pos", "action.wrist_l_rot6d", "action.wrist_r_pos", "action.wrist_r_rot6d"]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0, 8, 16, 24, 32]
    state_indices = [0]
    action_indices = list(range(32))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ])
    
class FourierGr1ArmsWaistNoActionDataConfig:
    video_keys = ["video.ego_view"]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0, 8, 16, 24, 32]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[])


ROBOT_TYPE_CONFIG_MAP = {
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "fourier_gr1_arms_waist_no_action": FourierGr1ArmsWaistNoActionDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "fourier_gr1_arms_waist": EmbodimentTag.GR1,
    "fourier_gr1_arms_waist_no_action": EmbodimentTag.NEW_EMBODIMENT_NO_ACTION,
}

DATASET_NAMED_MIXTURES = {
    "fourier_gr1_unified_1000": [
        ("gr1_unified.PnPBottleToCabinetClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCanToDrawerClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCupToDrawerClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPMilkToMicrowaveClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPPotatoToMicrowaveClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPWineToCabinetClose", 1.0, "fourier_gr1_arms_waist"),

        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA", 1.0, "fourier_gr1_arms_waist"),

        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA", 1.0, "fourier_gr1_arms_waist"),

        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA", 1.0, "fourier_gr1_arms_waist"),

        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA", 1.0, "fourier_gr1_arms_waist"),
    ],
}
