# Copyright 2026 WALA authors. All rights reserved.
# Licensed under the MIT License.

"""MLP action head used by WALA."""

import torch.nn as nn
import torch.nn.functional as F


class MLPResNetBlock(nn.Module):
    """One MLP residual block."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.ffn(x) + x


class MLPResNet(nn.Module):
    """MLP with residual blocks."""

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        return self.fc2(x)


class L1RegressionActionHead(nn.Module):
    """Continuous action regression head used by WALA."""

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=4096,
        action_dim=7,
        NUM_ACTIONS_CHUNK=8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states):
        batch_size, chunk_len, hidden_dim = actions_hidden_states.shape
        x = actions_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        x = self.model(x)
        return x.view(batch_size, chunk_len, self.action_dim)

    def forward(self, actions_hidden_states, actions=None):
        if actions is None:
            return self.predict_action(actions_hidden_states)
        if actions.shape[1] != self.NUM_ACTIONS_CHUNK:
            raise ValueError(
                f"Expected actions.shape[1] == NUM_ACTIONS_CHUNK={self.NUM_ACTIONS_CHUNK}, "
                f"but got actions.shape[1]={actions.shape[1]}."
            )
        return F.l1_loss(self.predict_action(actions_hidden_states), actions)


def get_action_model(config=None):
    """Build the MLP action head from ``config.framework.action_model``."""
    action_model_cfg = config.framework.action_model
    model_type = str(action_model_cfg.action_model_type).strip()
    if model_type != "MLP":
        raise ValueError(
            f"Unsupported action_model_type={model_type!r}; expected 'MLP'."
        )

    action_hidden_dim = int(action_model_cfg.action_hidden_dim)
    action_dim = int(action_model_cfg.action_dim)
    future_action_window_size = int(action_model_cfg.future_action_window_size)
    return L1RegressionActionHead(
        input_dim=action_hidden_dim,
        hidden_dim=action_hidden_dim * 2,
        action_dim=action_dim,
        NUM_ACTIONS_CHUNK=1 + future_action_window_size,
    )
