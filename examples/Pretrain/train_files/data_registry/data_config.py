"""Dataset configs, embodiment tags, and mixtures for WALA pretraining."""

import os
from pathlib import Path

from wala.dataloader.gr00t_lerobot.datasets import ModalityConfig
from wala.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from wala.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
    
class RoboCOINR1LiteNoActionDataConfig:
    video_keys = [
        "video.cam_high_rgb",
    ]

    language_keys = [
        "annotation.human.action.task_description"
    ]

    observation_indices = [0, 8, 16, 24, 32]

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[])

class RoboCOINCobotMagicNoActionDataConfig:
    video_keys = [
        "video.cam_high_rgb",
    ]

    language_keys = [
        "annotation.human.action.task_description"
    ]

    observation_indices = [0, 8, 16, 24, 32]

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[])


class RoboCOINShadowNoActionDataConfig(RoboCOINCobotMagicNoActionDataConfig):
    pass
    
class RobotwinNoActionConfig:
    video_keys = ["video.cam_high"]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0, 8, 16, 24, 32]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[])
    
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


class HumanVideoNoActionConfig:
    video_keys = ["video.cam_high_rgb"]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0, 8, 16, 24, 32]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[])


class LeRobotV3EgocentricNoActionConfig:
    video_keys = ["video.egocentric"]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0, 8, 16, 24, 32]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[])


ROBOT_TYPE_CONFIG_MAP = {
    "egodex_human_video": HumanVideoNoActionConfig(),
    "robocoin_r1_lite_no_action": RoboCOINR1LiteNoActionDataConfig(),
    "robocoin_cobot_magic_no_action": RoboCOINCobotMagicNoActionDataConfig(),
    "robocoin_shadow_no_action": RoboCOINShadowNoActionDataConfig(),
    "robotwin_no_action_of_pretrain": RobotwinNoActionConfig(),
    "robocasa_no_action": FourierGr1ArmsWaistNoActionDataConfig(),
    "lerobot_v3_egocentric_no_action": LeRobotV3EgocentricNoActionConfig(),
}


ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "egodex_human_video": EmbodimentTag.EGODEX_HUMAN_VIDEO,
    "robocoin_r1_lite_no_action": EmbodimentTag.ROBOCOIN_R1_LITE,
    "robocoin_cobot_magic_no_action": EmbodimentTag.ROBOCOIN_COBOT_MAGIC,
    "robocoin_shadow_no_action": EmbodimentTag.NEW_EMBODIMENT_NO_ACTION,
    "robotwin_no_action_of_pretrain": EmbodimentTag.NEW_EMBODIMENT_NO_ACTION,
    "robocasa_no_action": EmbodimentTag.GR1,
    "lerobot_v3_egocentric_no_action": EmbodimentTag.NEW_EMBODIMENT_NO_ACTION,
}


def _build_robocoin_shadow_mixture():
    root = Path(os.environ.get("WALA_ROBOCOIN_SHADOW_ROOT", "/mnt/data/ljh/WALA/cache/robocoin/RoboCOIN"))
    if not root.exists():
        return []
    dataset_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "meta" / "modality.json").exists()
    ]
    return [(str(path), 1.0, "robocoin_shadow_no_action") for path in dataset_dirs]


DATASET_NAMED_MIXTURES = {
    "robocoin_shadow_no_action": _build_robocoin_shadow_mixture(),
    "pretrain_no_action": [
        ("data/ljh/egodex_data_lerobot/part1_dataset", 0.20000000, "egodex_human_video"),
        ("data/ljh/egodex_data_lerobot/part2_dataset", 0.20000000, "egodex_human_video"),
        ("data/ljh/egodex_data_lerobot/part3_dataset", 0.20000000, "egodex_human_video"),
        ("data/ljh/egodex_data_lerobot/part4_dataset", 0.20000000, "egodex_human_video"),
        ("data/ljh/egodex_data_lerobot/part5_dataset", 0.20000000, "egodex_human_video"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_catch_the_ball", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_classification_of_tableware", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_clear_the_desktop", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_close_book", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_close_button", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_fold_the_towel", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_fold_towel_a", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_move_the_bread", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_move_the_cup", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_move_the_plate", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_move_the_small_ball", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_movethe_position_of_the_bluetooth", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_place_the_test_tube", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_plate_storage_apple", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_plate_storage_bread", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_plate_storaje_baozi", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_put_in_the_pear", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_put_the_building_block_on_the_table", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_steamer_storage_dumpling", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_storage_plate", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_take_out_a_pen_from_the_pen_holder", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_take_the_shoes_off_the_shelf", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_the_box_stores_table_tennis_balls", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_the_plate_holds_the_fruit", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_the_plate_holds_the_vegetables", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_turn_on_the_bulb", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/Cobot_Magic_turn_on_the_desk_lamp", 0.03703704, "robocoin_cobot_magic_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_boil_water_in_a_kettle", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_catch_the_water", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_clean_the_floor", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_clean_the_sink", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_clean_toilet", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_connect_the_router_cable", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_cover_the_pot_lid", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_dispose_of_leftover_food", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_fold_clothes", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_garbage_disposal", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_make_a_landline_call", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_make_breakfast", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_make_the_bed", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_open_and_close_curtains", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_open_and_close_nightstand_door", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_open_and_close_nightstand_drawer", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_open_and_close_the_freezer_door", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_open_the_food_pan", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_opening_and_closing_aalcony_sliding_doors", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_place_the_dress_shirt_on_the_hanger", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_pour_water", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_put_on_a_garbage_bag", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_put_the_pillow_on_the_bed", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_put_the_shoes_into_the_shoe_box", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_sliding_chair", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_storage_of_toiletries", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_switch_labels", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_switch_on_and_off_the_central_air_conditioning", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_tableware_arrangement", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_tableware_cleaning", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_take_and_place_the_portable_power_bank", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_take_and_put_away_garden_stuff", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_take_and_put_away_items", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_take_and_put_the_bowl", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_take_or_store_plates", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_tea_service_table_setting", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_tidy_up_toiletries", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_wash_the_tableware", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_washing_board", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robocoin_dataset_local_delta29/R1_Lite_wipe_the_table", 0.02500000, "robocoin_r1_lite_no_action"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/adjust_bottle-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/adjust_bottle-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/beat_block_hammer-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/beat_block_hammer-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/blocks_ranking_rgb-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/blocks_ranking_rgb-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/blocks_ranking_size-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/blocks_ranking_size-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/click_alarmclock-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/click_alarmclock-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/click_bell-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/click_bell-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/dump_bin_bigbin-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/dump_bin_bigbin-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/grab_roller-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/grab_roller-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/handover_block-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/handover_block-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/handover_mic-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/handover_mic-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/hanging_mug-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/hanging_mug-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/lift_pot-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/lift_pot-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/move_can_pot-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/move_can_pot-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/move_pillbottle_pad-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/move_pillbottle_pad-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/move_playingcard_away-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/move_playingcard_away-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/move_stapler_pad-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/move_stapler_pad-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/open_laptop-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/open_laptop-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/open_microwave-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/open_microwave-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/pick_diverse_bottles-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/pick_diverse_bottles-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/pick_dual_bottles-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/pick_dual_bottles-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_a2b_left-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_a2b_left-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_a2b_right-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_a2b_right-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_bread_basket-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_bread_basket-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_bread_skillet-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_bread_skillet-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_burger_fries-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_burger_fries-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_can_basket-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_can_basket-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_cans_plasticbox-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_cans_plasticbox-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_container_plate-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_container_plate-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_dual_shoes-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_dual_shoes-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_empty_cup-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_empty_cup-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_fan-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_fan-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_mouse_pad-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_mouse_pad-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_object_basket-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_object_basket-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_object_scale-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_object_scale-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_object_stand-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_object_stand-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_phone_stand-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_phone_stand-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_shoe-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/place_shoe-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/press_stapler-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/press_stapler-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/put_bottles_dustbin-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/put_bottles_dustbin-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/put_object_cabinet-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/put_object_cabinet-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/rotate_qrcode-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/rotate_qrcode-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/scan_object-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/scan_object-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/shake_bottle-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/shake_bottle-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/shake_bottle_horizontally-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/shake_bottle_horizontally-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/stack_blocks_three-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/stack_blocks_three-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/stack_blocks_two-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/stack_blocks_two-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/stack_bowls_three-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/stack_bowls_three-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/stack_bowls_two-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/stack_bowls_two-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/stamp_seal-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/stamp_seal-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/turn_switch-aloha-agilex_clean_50-50", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/ljh/robotwin_dataset/lerobot_abs_qpos_rgb/turn_switch-aloha-agilex_randomized_500-500", 0.01000000, "robotwin_no_action_of_pretrain"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PnPBottleToCabinetClose", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PnPCanToDrawerClose", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PnPCupToDrawerClose", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PnPMilkToMicrowaveClose", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PnPPotatoToMicrowaveClose", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PnPWineToCabinetClose", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA", 0.04166667, "robocasa_no_action"),
        ("data/lhr/PhysicalAI-Robotics-GR00T-Teleop-Sim_LeRobot-AugPosRot-Correct_lerobot_v21/gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA", 0.04166667, "robocasa_no_action"),
    ],

}
