from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler
from wala.model.framework.share_tools import read_mode_config


class PolicyWarper:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10095,
        n_action_steps=2,
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.n_action_steps = n_action_steps

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats, self.state_norm_stats = self.get_action_and_state_stats(
            self.unnorm_key,
            policy_ckpt_path=policy_ckpt_path,
        )

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str or tuple) -> None:

        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(self, observations, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Execute one inference step.
        :param image: Input image (H, W, 3) in uint8 format
        :param task_description: Task description text
        :return: (raw_actions, processed_actions)
        """
        task_description = observations["annotation.human.coarse_action"][0]  # tuple

        ego_view = observations["video.ego_view"]  # (N, 1, H, W, 3)
        images = ego_view
        state = {}

        state["left_arm"] = observations["state.left_arm"]      # (N, 1, 7)
        state["right_arm"] = observations["state.right_arm"]    # (N, 1, 7)
        state["left_hand"] = observations["state.left_hand"]    # (N, 1, 6)
        state["right_hand"] = observations["state.right_hand"]  # (N, 1, 6)
        state["waist"] = observations["state.waist"]            # (N, 1, 3)

        state["left_wrist_pos"] = observations["state.wrist_l_pos"]        # (N, 1, 3)
        state["left_wrist_rot6d"] = observations["state.wrist_l_rot6d"]    # (N, 1, 6)
        state["right_wrist_pos"] = observations["state.wrist_r_pos"]       # (N, 1, 3)
        state["right_wrist_rot6d"] = observations["state.wrist_r_rot6d"]   # (N, 1, 6)

        state_order = [
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "waist",
            "left_wrist_pos",
            "left_wrist_rot6d",
            "right_wrist_pos",
            "right_wrist_rot6d",
        ]

        input_state = np.concatenate([state[k] for k in state_order], axis=-1)  # (N, 1, 47)
        input_state = self.normalize_state(input_state, self.state_norm_stats)  # (N, 1, 47)

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [[self._resize_image(img) for img in sample] for sample in images]  # (B, N_view, H, W, 3)
        input_state = [input_s for input_s in input_state]  # B, 1, 47

        examples = []
        batch_size = len(images)
        instructions = [self.task_description] if isinstance(self.task_description, str) else self.task_description
   
        for b in range(batch_size):
            vla_formatted_images = [[ [img], [] ] for img in images[b]]
            
            example = {
                "image": vla_formatted_images,
                "lang": instructions[b] if isinstance(instructions, list) else instructions,
                "state": input_state[b].reshape(1, -1),
            }
            examples.append(example)

        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        response = self.client.predict_action(vla_input)

        normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D

        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
        )

        # raw_actions shape: (B, chunk, D)
        if self.action_ensemble:
            # Ensemble each sample in the batch
            batch_size = raw_actions.shape[0]
            ensembled_actions = []
            for b in range(batch_size):
                ensembled = self.action_ensembler.ensemble_action(raw_actions[b])[None]  # (1, D)
                ensembled_actions.append(ensembled)
            raw_actions = np.stack(ensembled_actions, axis=0)  # (B, 1, D)

        raw_action = {
            "action.left_arm": raw_actions[:, : self.n_action_steps, :7],  # (B, n_action_steps, 7)
            "action.right_arm": raw_actions[:, : self.n_action_steps, 7:14],  # (B, n_action_steps, 7)
            "action.left_hand": raw_actions[:, : self.n_action_steps, 14:20],  # (B, n_action_steps, 6)
            "action.right_hand": raw_actions[:, : self.n_action_steps, 20:26],  # (B, n_action_steps, 6)
            "action.waist": raw_actions[:, : self.n_action_steps, 26:29],  # (B, n_action_steps, 3)
        }

        return {"actions": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Args:
            normalized_actions: shape (B, chunk, D) (chunk, D)
            action_norm_stats:
        Returns:
            actions
        """
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])

        normalized_actions = np.clip(normalized_actions, -1, 1)

        actions = np.where(
            mask,
            (normalized_actions + 1) / 2 * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_and_state_stats(unnorm_key: str, policy_ckpt_path) -> tuple[dict, dict]:
        """
        Load action and state normalization statistics from checkpoint config.
        Expected:
            norm_stats[unnorm_key]["action"]["min/max/mask"]
            norm_stats[unnorm_key]["state"]["min/max/mask"]
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)

        unnorm_key = PolicyWarper._check_unnorm_key(norm_stats, unnorm_key)
        dataset_stats = norm_stats[unnorm_key]

        if "action" not in dataset_stats:
            raise KeyError(
                f"`action` statistics not found in norm_stats[{unnorm_key}]. "
                f"Available keys: {dataset_stats.keys()}"
            )

        if "state" not in dataset_stats:
            raise KeyError(
                f"`state` statistics not found in norm_stats[{unnorm_key}]. "
                f"Available keys: {dataset_stats.keys()}. "
                "If your checkpoint was trained without state min/max stats, "
                "you need to recompute/save 47-dim state statistics first."
            )

        return dataset_stats["action"], dataset_stats["state"]

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """Validate and resolve the dataset statistics key."""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @staticmethod
    def normalize_state(state: np.ndarray, state_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Min-max normalize state to [-1, 1].

        Args:
            state: shape (B, T, D), here D should be 47.
            state_norm_stats: dict with "min", "max", optionally "mask".

        Returns:
            normalized_state: shape (B, T, D)
        """
        state_min = np.asarray(state_norm_stats["min"], dtype=np.float32)
        state_max = np.asarray(state_norm_stats["max"], dtype=np.float32)
        mask = np.asarray(
            state_norm_stats.get("mask", np.ones_like(state_min, dtype=bool)),
            dtype=bool,
        )

        state = state.astype(np.float32)

        # Make sure stats can broadcast to (B, T, D)
        state_min = state_min.reshape(1, 1, -1)
        state_max = state_max.reshape(1, 1, -1)
        mask = mask.reshape(1, 1, -1)

        denom = state_max - state_min
        denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)

        normalized_state = 2.0 * (state - state_min) / denom - 1.0
        normalized_state = np.clip(normalized_state, -1.0, 1.0)

        # If mask=False, keep raw value, consistent with action unnormalization style.
        normalized_state = np.where(mask, normalized_state, state)

        return normalized_state
