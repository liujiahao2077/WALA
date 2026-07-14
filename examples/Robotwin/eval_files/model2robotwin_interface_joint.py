from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from wala.model.tools import read_mode_config

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class ObservationBuffer:
    def __init__(self, max_history=64):
        self.max_history = max_history
        self.head_history = []
        self.left_history = []
        self.right_history = []

    def reset(self, initial_obs):
        """Fill the history buffer with the first observation."""
        head = initial_obs["observation"]["head_camera"]["rgb"]
        left = initial_obs["observation"]["left_camera"]["rgb"]
        right = initial_obs["observation"]["right_camera"]["rgb"]
        
        self.head_history = [head] * self.max_history
        self.left_history = [left] * self.max_history
        self.right_history = [right] * self.max_history

    def add(self, obs):
        """Append a new observation to the history buffer."""
        self.head_history.append(obs["observation"]["head_camera"]["rgb"])
        self.left_history.append(obs["observation"]["left_camera"]["rgb"])
        self.right_history.append(obs["observation"]["right_camera"]["rgb"])
        
        if len(self.head_history) > self.max_history * 2:
            self.head_history = self.head_history[-self.max_history:]
            self.left_history = self.left_history[-self.max_history:]
            self.right_history = self.right_history[-self.max_history:]

    def get_frame_sequence(self, indices):
        """Return per-camera frame sequences for the requested history indices."""
        head_seq = [self.head_history[i] for i in indices]
        left_seq = [self.left_history[i] for i in indices]
        right_seq = [self.right_history[i] for i in indices]
        return [[head_seq, []], [left_seq, []], [right_seq, []]]

class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        self.task_description = None
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path, action_mode=action_mode
        )
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.raw_actions = None

        self.obs_buffer = ObservationBuffer(max_history=64)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None

    def step(self, example: dict, step: int = 0) -> np.ndarray:
        state = example.get("state", None)
        if state is not None:
            state = self.normalize_state(state, self.state_norm_stats)
            example["state"] = state.reshape(1, -1)

        task_description = example.get("lang", None)
        images = example["image"]

        if task_description != self.task_description or step == 0:
            self.reset(task_description)

        def recursive_resize(img):
            if isinstance(img, list):
                return [recursive_resize(i) for i in img]
            return self._resize_image(img)

        images = [recursive_resize(img) for img in images]
        example["image"] = images
        example_copy = example.copy()

        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"][0]  # B, chunk, D -> chunk, D
                action_dim = np.array(self.action_norm_stats.get("max")).shape[0]
                normalized_actions = normalized_actions[:, :action_dim]
                normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
            except KeyError:
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
                normalization_mode=self.normalization_mode,
            )

        action_idx = step % self.action_chunk_size
        current_action = self.raw_actions[action_idx]

        return current_action

    @staticmethod
    def normalize_state(state: np.ndarray, state_norm_stats: Dict[str, np.ndarray], normalization_mode: str = "min_max") -> dict[str, np.ndarray]:
        dim = state.shape[-1]
        continuous_mask = np.ones(dim, dtype=bool)
        state_high, state_low = ModelClient._get_normalization_bounds(state_norm_stats, normalization_mode=normalization_mode)
        valid_mask = continuous_mask & (state_high != state_low)
        normalized_state = np.where(valid_mask, (state - state_low) / (state_high - state_low) * 2 - 1, state)
        normalized_state = np.where(valid_mask, np.clip(normalized_state, -1.0, 1.0), normalized_state)
        normalized_state = np.where(~continuous_mask, (normalized_state > 0.5).astype(normalized_state.dtype), normalized_state)
        return normalized_state

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray], normalization_mode: str = "min_max") -> np.ndarray:
        action_high, action_low = ModelClient._get_normalization_bounds(action_norm_stats, normalization_mode=normalization_mode)
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
        normalized_actions = np.clip(normalized_actions, -1, 1)
        actions = np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path, action_mode: str = "abs") -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        stats = norm_stats[unnorm_key]
        if action_mode in stats:
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        if "action" in stats:
            return stats["action"]
        raise ValueError(f"Invalid statistics file format for key `{unnorm_key}`.")

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        available_keys = sorted(norm_stats.keys())
        if unnorm_key is None:
            if len(available_keys) == 1:
                return available_keys[0]
            raise ValueError(f"`unnorm_key` must be provided. Available keys: {available_keys}")
        if unnorm_key not in norm_stats:
            raise KeyError(f"Unknown `unnorm_key`: `{unnorm_key}`. Available keys: {available_keys}")
        return unnorm_key

    @staticmethod
    def _get_normalization_bounds(norm_stats: Dict[str, np.ndarray], normalization_mode: str = "min_max") -> tuple[np.ndarray, np.ndarray]:
        if normalization_mode == "q99":
            return np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        if normalization_mode == "min_max":
            return np.array(norm_stats["max"]), np.array(norm_stats["min"])
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get("action_normalization_mode", usr_args.get("normalization_mode", "min_max"))

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
    )

def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    if TASK_ENV.take_action_cnt == 0:
        model.obs_buffer.reset(observation)
    else:
        model.obs_buffer.add(observation)
    
    indices = [-1]
    images_sequence = model.obs_buffer.get_frame_sequence(indices)

    instruction = TASK_ENV.get_instruction()

    left_arm_joints = observation["joint_action"]["left_arm"]
    left_gripper = observation["joint_action"]["left_gripper"]
    right_arm_joints = observation["joint_action"]["right_arm"]
    right_gripper = observation["joint_action"]["right_gripper"]

    qpos_state = np.concatenate([
        left_arm_joints, [left_gripper], 
        right_arm_joints, [right_gripper]
    ])

    example = {
        "lang": str(instruction),
        "image": images_sequence, 
        "state": qpos_state, 
    }

    joint_action = model.step(example, step=TASK_ENV.take_action_cnt)

    TASK_ENV.take_action(joint_action, action_type="qpos")
