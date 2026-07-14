"""RobotWin benchmark — data config, embodiment tags, and mixtures."""

from wala.dataloader.gr00t_lerobot.datasets import ModalityConfig
from wala.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from wala.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from wala.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig — Agilex (RobotWin)
# ---------------------------------------------------------------------------
class AgilexDataConfig:
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.states"]
    action_keys = ["action.actions"]
    language_keys = ["annotation.human.action.task_description"]
    # observation_indices = [-32, -16, 0, 10, 21, 32]
    # observation_indices = [-32, -16, 0]
    # observation_indices = [0, 10, 20, 30, 40, 50]
    observation_indices = [0, 8, 16, 24, 32]
    # observation_indices = [0]
    # observation_indices = [-16, -12, -8, -4, 0, 16]
    state_indices = [0]
    # action_indices = list(range(-16, 16))
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
                normalization_modes={
                    "state.states": "min_max",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.actions": "min_max",
                },
            ),
        ])
    
# ---------------------------------------------------------------------------
# DataConfig — Agilex (RobotWin No Action)
# ---------------------------------------------------------------------------
class AgilexDataNoActionConfig:
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    # state_keys = ["state.states"]
    # action_keys = ["action.actions"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0,10,20,30,40,50]
    # observation_indices = [0]
    # state_indices = []
    # action_indices = []

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            # "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            # "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[])

ROBOT_TYPE_CONFIG_MAP = {
    "robotwin": AgilexDataConfig(),
    "robotwin_no_action": AgilexDataNoActionConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Uses NEW_EMBODIMENT fallback (not in base embodiment_tags.py)
    "robotwin": EmbodimentTag.NEW_EMBODIMENT,
    "robotwin_no_action": EmbodimentTag.NEW_EMBODIMENT_NO_ACTION,
}

# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "robotwin_test": [("beat_block_hammer-aloha-agilex_clean_50-50", 1.0, "robotwin")],

    "robotwin_all": [
        ("adjust_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("adjust_bottle-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("beat_block_hammer-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("beat_block_hammer-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("blocks_ranking_rgb-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("blocks_ranking_rgb-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("blocks_ranking_size-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("blocks_ranking_size-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("click_alarmclock-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("click_alarmclock-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("click_bell-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("click_bell-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("dump_bin_bigbin-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("dump_bin_bigbin-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("grab_roller-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("grab_roller-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("handover_block-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("handover_block-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("handover_mic-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("handover_mic-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("hanging_mug-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("hanging_mug-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("lift_pot-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("lift_pot-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("move_can_pot-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("move_can_pot-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("move_pillbottle_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("move_pillbottle_pad-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("move_playingcard_away-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("move_playingcard_away-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("move_stapler_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("move_stapler_pad-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("open_laptop-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("open_laptop-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("open_microwave-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("open_microwave-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("pick_diverse_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("pick_diverse_bottles-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("pick_dual_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("pick_dual_bottles-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_a2b_left-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_a2b_left-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_a2b_right-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_a2b_right-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_bread_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_bread_basket-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_bread_skillet-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_bread_skillet-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_burger_fries-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_burger_fries-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_can_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_can_basket-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_cans_plasticbox-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_cans_plasticbox-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_container_plate-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_container_plate-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_dual_shoes-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_dual_shoes-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_empty_cup-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_empty_cup-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_fan-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_fan-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_mouse_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_mouse_pad-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_object_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_object_basket-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_object_scale-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_object_scale-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_object_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_object_stand-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_phone_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_phone_stand-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("place_shoe-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_shoe-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("press_stapler-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("press_stapler-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("put_bottles_dustbin-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("put_bottles_dustbin-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("put_object_cabinet-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("put_object_cabinet-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("rotate_qrcode-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("rotate_qrcode-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("scan_object-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("scan_object-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("shake_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("shake_bottle-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("shake_bottle_horizontally-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("shake_bottle_horizontally-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("stack_blocks_three-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("stack_blocks_three-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("stack_blocks_two-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("stack_blocks_two-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("stack_bowls_three-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("stack_bowls_three-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("stack_bowls_two-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("stack_bowls_two-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("stamp_seal-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("stamp_seal-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
        ("turn_switch-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("turn_switch-aloha-agilex_randomized_500-500", 1.0, "robotwin"),
    ],

    "robotwin_clean": [
        ("adjust_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("beat_block_hammer-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("blocks_ranking_rgb-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("blocks_ranking_size-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("click_alarmclock-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("click_bell-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("dump_bin_bigbin-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("grab_roller-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("handover_block-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("handover_mic-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("hanging_mug-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("lift_pot-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("move_can_pot-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("move_pillbottle_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("move_playingcard_away-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("move_stapler_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("open_laptop-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("open_microwave-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("pick_diverse_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("pick_dual_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_a2b_left-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_a2b_right-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_bread_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_bread_skillet-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_burger_fries-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_can_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_cans_plasticbox-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_container_plate-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_dual_shoes-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_empty_cup-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_fan-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_mouse_pad-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_object_basket-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_object_scale-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_object_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_phone_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_shoe-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("press_stapler-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("put_bottles_dustbin-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("put_object_cabinet-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("rotate_qrcode-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("scan_object-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("shake_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("shake_bottle_horizontally-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("stack_blocks_three-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("stack_blocks_two-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("stack_bowls_three-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("stack_bowls_two-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("stamp_seal-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("turn_switch-aloha-agilex_clean_50-50", 1.0, "robotwin"),
    ],

    "robotwin_clean_8":[
        ("adjust_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("beat_block_hammer-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("dump_bin_bigbin-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("grab_roller-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("lift_pot-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("move_playingcard_away-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("pick_dual_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_object_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"),
    ],

    "robotwin_clean_8_demo10":[
        ("lerobot_abs_qpos_new_data_10/adjust_bottle-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/beat_block_hammer-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/dump_bin_bigbin-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/grab_roller-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/lift_pot-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/move_playingcard_away-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/pick_dual_bottles-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/place_object_stand-demo_clean-50", 1.0, "robotwin"),
    ],

    "robotwin_clean_8_demo10_mix_no_action":[
        ("lerobot_abs_qpos_new_data_10/adjust_bottle-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/beat_block_hammer-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/dump_bin_bigbin-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/grab_roller-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/lift_pot-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/move_playingcard_away-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/pick_dual_bottles-demo_clean-50", 1.0, "robotwin"),
        ("lerobot_abs_qpos_new_data_10/place_object_stand-demo_clean-50", 1.0, "robotwin"),
        
        ("adjust_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"),
        ("beat_block_hammer-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"),
        ("dump_bin_bigbin-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"),
        ("grab_roller-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"),
        ("lift_pot-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"),
        ("move_playingcard_away-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"),
        ("pick_dual_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"),
        ("place_object_stand-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"),

        ("adjust_bottle-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("beat_block_hammer-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("dump_bin_bigbin-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("grab_roller-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("lift_pot-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("move_playingcard_away-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("pick_dual_bottles-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_object_stand-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
    ],
    
    "robotwin_clean_4":[
        ("adjust_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("pick_dual_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("place_object_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"),
        ("move_playingcard_away-aloha-agilex_clean_50-50", 1.0, "robotwin"),
    ],

    "robotwin_clean_4_mix_no_action_randomized": [
        ("adjust_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("adjust_bottle-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("pick_dual_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("pick_dual_bottles-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_object_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("place_object_stand-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("move_playingcard_away-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("move_playingcard_away-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
    ],

    "robotwin_clean_4_mix_no_action_clean": [
        ("adjust_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("lerobot_abs_qpos_new_data/adjust_bottle-demo_clean-50", 1.0, "robotwin_no_action"),
        ("pick_dual_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("lerobot_abs_qpos_new_data/pick_dual_bottles-demo_clean-50", 1.0, "robotwin_no_action"),
        ("place_object_stand-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("lerobot_abs_qpos_new_data/place_object_stand-demo_clean-50", 1.0, "robotwin_no_action"),
        ("move_playingcard_away-aloha-agilex_clean_50-50", 1.0, "robotwin"), ("lerobot_abs_qpos_new_data/move_playingcard_away-demo_clean-50", 1.0, "robotwin_no_action"),
    ],


    "robotwin_all_no_action": [
        ("adjust_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("adjust_bottle-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("beat_block_hammer-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("beat_block_hammer-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("blocks_ranking_rgb-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("blocks_ranking_rgb-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("blocks_ranking_size-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("blocks_ranking_size-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("click_alarmclock-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("click_alarmclock-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("click_bell-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("click_bell-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("dump_bin_bigbin-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("dump_bin_bigbin-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("grab_roller-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("grab_roller-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("handover_block-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("handover_block-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("handover_mic-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("handover_mic-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("hanging_mug-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("hanging_mug-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("lift_pot-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("lift_pot-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("move_can_pot-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("move_can_pot-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("move_pillbottle_pad-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("move_pillbottle_pad-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("move_playingcard_away-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("move_playingcard_away-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("move_stapler_pad-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("move_stapler_pad-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("open_laptop-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("open_laptop-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("open_microwave-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("open_microwave-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("pick_diverse_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("pick_diverse_bottles-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("pick_dual_bottles-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("pick_dual_bottles-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_a2b_left-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_a2b_left-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_a2b_right-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_a2b_right-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_bread_basket-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_bread_basket-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_bread_skillet-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_bread_skillet-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_burger_fries-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_burger_fries-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_can_basket-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_can_basket-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_cans_plasticbox-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_cans_plasticbox-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_container_plate-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_container_plate-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_dual_shoes-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_dual_shoes-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_empty_cup-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_empty_cup-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_fan-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_fan-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_mouse_pad-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_mouse_pad-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_object_basket-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_object_basket-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_object_scale-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_object_scale-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_object_stand-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_object_stand-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_phone_stand-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_phone_stand-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("place_shoe-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("place_shoe-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("press_stapler-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("press_stapler-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("put_bottles_dustbin-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("put_bottles_dustbin-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("put_object_cabinet-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("put_object_cabinet-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("rotate_qrcode-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("rotate_qrcode-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("scan_object-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("scan_object-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("shake_bottle-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("shake_bottle-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("shake_bottle_horizontally-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("shake_bottle_horizontally-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("stack_blocks_three-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("stack_blocks_three-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("stack_blocks_two-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("stack_blocks_two-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("stack_bowls_three-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("stack_bowls_three-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("stack_bowls_two-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("stack_bowls_two-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("stamp_seal-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("stamp_seal-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
        ("turn_switch-aloha-agilex_clean_50-50", 1.0, "robotwin_no_action"), ("turn_switch-aloha-agilex_randomized_500-500", 1.0, "robotwin_no_action"),
    ],
}
