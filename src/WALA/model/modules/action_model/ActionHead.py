# Copyright 2025 WALA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Action heads for OFT-style VLA models.

This version replaces the DiT/cross-attention flow head with an OFT-friendly
interleaved-token flow decoder:

    [time, optional_state, q1, x1, q2, x2, ..., qT, xT]

where q_i is the i-th latent action query and x_i is the noised action token.
The optional state token is prepended before the interleaved query/action tokens.
The noised action tokens predict flow velocity, while query tokens receive an
auxiliary clean-action supervision signal.

The optional self-attention mask keeps the condition side clean:
    - time/state/query tokens cannot attend to noised action tokens.
    - noised action tokens can attend to all tokens.

Set use_attention_mask=False to recover the original full self-attention
behavior and remain compatible with old model weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta



class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal timestep embedding used by the flow action head."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()
        _, _ = timesteps.shape
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0, device=device)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        if enc.shape[-1] < self.embedding_dim:
            enc = torch.nn.functional.pad(enc, (0, self.embedding_dim - enc.shape[-1]))
        return enc


# ============================================================
# MLP Action Head
# ============================================================

class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)

        for block in self.mlp_resnet_blocks:
            x = block(x)

        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via regression."""

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
        """
        Args:
            actions_hidden_states:
                Shape: (B, chunk_len, hidden_dim)

        Returns:
            actions:
                Shape: (B, chunk_len, action_dim)
        """

        batch_size, chunk_len, hidden_dim = actions_hidden_states.shape
        x = actions_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        x = self.model(x)
        actions = x.view(batch_size, chunk_len, self.action_dim)
        return actions

    def forward(self, actions_hidden_states, actions=None):
        if actions is not None:
            if actions.shape[1] != self.NUM_ACTIONS_CHUNK:
                raise ValueError(
                    f"Expected actions.shape[1] == NUM_ACTIONS_CHUNK={self.NUM_ACTIONS_CHUNK}, "
                    f"but got actions.shape[1]={actions.shape[1]}."
                )
            loss = F.l1_loss(
                self.predict_action(actions_hidden_states),
                actions,
            )
            return loss
        return self.predict_action(actions_hidden_states)


class TransformerRegressionActionHead(nn.Module):
    """Transformer-based chunk action regression head.

    This is an independent alternative to the MLP action head. It treats OFT
    action queries as aligned latent action tokens and applies bidirectional
    self-attention across the whole action chunk. Optional state and external
    condition tokens can be prepended before the action tokens.
    """

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=None,
        action_dim=7,
        NUM_ACTIONS_CHUNK=8,
        num_layers=4,
        num_heads=16,
        ffn_multiplier=4,
        dropout=0.0,
        state_dim=0,
        condition_dim=0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.input_dim
        self.action_dim = int(action_dim)
        self.NUM_ACTIONS_CHUNK = int(NUM_ACTIONS_CHUNK)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_multiplier = int(ffn_multiplier)
        self.state_dim = int(state_dim) if state_dim is not None else 0
        self.condition_dim = int(condition_dim) if condition_dim is not None else 0

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"Transformer hidden_dim={self.hidden_dim} must be divisible by "
                f"num_heads={self.num_heads}."
            )

        self.action_query_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
        )

        if self.state_dim > 0:
            self.state_proj = nn.Sequential(
                nn.LayerNorm(self.state_dim),
                nn.Linear(self.state_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.state_proj = None

        if self.condition_dim > 0:
            self.condition_proj = nn.Sequential(
                nn.LayerNorm(self.condition_dim),
                nn.Linear(self.condition_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.condition_proj = None

        self.step_pos = nn.Embedding(self.NUM_ACTIONS_CHUNK, self.hidden_dim)
        # 0: state, 1: external condition, 2: action query
        self.type_embedding = nn.Embedding(3, self.hidden_dim)
        nn.init.normal_(self.step_pos.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * self.ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def _build_tokens(self, actions_hidden_states, states=None, condition_tokens=None):
        B, T, _ = actions_hidden_states.shape
        device = actions_hidden_states.device
        dtype = actions_hidden_states.dtype

        if T != self.NUM_ACTIONS_CHUNK:
            raise ValueError(
                f"Expected actions_hidden_states.shape[1] == NUM_ACTIONS_CHUNK="
                f"{self.NUM_ACTIONS_CHUNK}, but got {T}."
            )

        action_tokens = self.action_query_proj(actions_hidden_states)
        pos_ids = torch.arange(T, dtype=torch.long, device=device)
        action_tokens = action_tokens + self.step_pos(pos_ids).unsqueeze(0).to(dtype=dtype)
        action_tokens = action_tokens + self.type_embedding.weight[2].view(1, 1, -1).to(dtype=dtype)

        prefix_tokens = []

        if states is not None and self.state_proj is not None:
            if states.ndim == 3:
                states = states[:, -1, :]
            state_token = self.state_proj(states.to(device=device, dtype=dtype)).unsqueeze(1)
            state_token = state_token + self.type_embedding.weight[0].view(1, 1, -1).to(dtype=dtype)
            prefix_tokens.append(state_token)

        if condition_tokens is not None and self.condition_proj is not None:
            condition_tokens = self.condition_proj(condition_tokens.to(device=device, dtype=dtype))
            condition_tokens = condition_tokens + self.type_embedding.weight[1].view(1, 1, -1).to(dtype=dtype)
            prefix_tokens.append(condition_tokens)

        if prefix_tokens:
            tokens = torch.cat(prefix_tokens + [action_tokens], dim=1)
            action_start = tokens.shape[1] - T
        else:
            tokens = action_tokens
            action_start = 0

        return tokens, action_start

    def predict_action(
        self,
        actions_hidden_states,
        states=None,
        condition_tokens=None,
    ):
        tokens, action_start = self._build_tokens(
            actions_hidden_states=actions_hidden_states,
            states=states,
            condition_tokens=condition_tokens,
        )
        tokens = self.transformer(tokens)
        action_tokens = tokens[:, action_start: action_start + self.NUM_ACTIONS_CHUNK]
        return self.action_head(action_tokens)

    def forward(
        self,
        actions_hidden_states,
        actions=None,
        states=None,
        condition_tokens=None,
        **kwargs,
    ):
        pred_actions = self.predict_action(
            actions_hidden_states=actions_hidden_states,
            states=states,
            condition_tokens=condition_tokens,
        )

        if actions is None:
            return pred_actions

        if actions.shape[1] != self.NUM_ACTIONS_CHUNK:
            raise ValueError(
                f"Expected actions.shape[1] == NUM_ACTIONS_CHUNK={self.NUM_ACTIONS_CHUNK}, "
                f"but got actions.shape[1]={actions.shape[1]}."
            )

        return F.l1_loss(pred_actions, actions)


# ============================================================
# Interleaved Token Flow Matching Action Head
# ============================================================
class InterleavedFlowDecoder(nn.Module):
    """Self-attention decoder over interleaved latent-query and noised-action tokens."""

    def __init__(
        self,
        action_hidden_dim,
        action_dim,
        action_horizon,
        state_dim=0,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
        dropout=0.0,
        use_attention_mask=False,
    ):
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.state_dim = int(state_dim) if state_dim is not None else 0
        self.hidden_dim = int(hidden_dim)
        self.use_attention_mask = bool(use_attention_mask)

        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.action_hidden_dim),
            nn.Linear(self.action_hidden_dim, self.hidden_dim),
        )
        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.hidden_dim),
        )
        if self.state_dim > 0:
            self.state_proj = nn.Sequential(
                nn.LayerNorm(self.state_dim),
                nn.Linear(self.state_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.state_proj = None

        self.time_encoder = SinusoidalPositionalEncoding(self.hidden_dim)
        self.time_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Shared step positions explicitly say q_i and x_i belong to the same
        # action timestep. Type embeddings distinguish time/query/action tokens.
        self.step_pos = nn.Embedding(self.action_horizon, self.hidden_dim)
        self.type_embedding = nn.Embedding(4, self.hidden_dim)
        nn.init.normal_(self.step_pos.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.velocity_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.query_action_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def _build_attention_mask(self, prefix_len, action_horizon, device):
        """
        Build a boolean self-attention mask for:

            [prefix..., q1, x1, q2, x2, ..., qT, xT]

        PyTorch TransformerEncoder bool mask convention:
            True means the attention edge is blocked.

        Mask rule:
            - prefix tokens and query tokens do not attend to noised action tokens.
            - noised action tokens attend to everything.

        This keeps q/time/state free of x_t noise leakage, while preserving the
        gradient path from flow loss to q through x-token attention to q.
        """
        seq_len = prefix_len + 2 * action_horizon
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        interleaved_start = prefix_len
        query_indices = interleaved_start + torch.arange(
            0,
            2 * action_horizon,
            2,
            dtype=torch.long,
            device=device,
        )
        action_indices = interleaved_start + torch.arange(
            1,
            2 * action_horizon,
            2,
            dtype=torch.long,
            device=device,
        )

        condition_rows = torch.cat(
            [
                torch.arange(prefix_len, dtype=torch.long, device=device),
                query_indices,
            ],
            dim=0,
        )

        mask[condition_rows[:, None], action_indices[None, :]] = True
        return mask

    def forward(self, action_queries, noised_actions, timesteps, states=None):
        """
        Args:
            action_queries:
                Shape: (B, T, action_hidden_dim)

            noised_actions:
                Shape: (B, T, action_dim)

            timesteps:
                Shape: (B,), discrete long timesteps.

            states:
                Optional. Shape: (B, state_dim) or (B, state_seq, state_dim).

        Returns:
            pred_velocity:
                Shape: (B, T, action_dim), predicted velocity from action tokens.

            pred_query_actions:
                Shape: (B, T, action_dim), auxiliary clean action from query tokens.
        """
        B, T, _ = noised_actions.shape
        device = noised_actions.device
        dtype = noised_actions.dtype

        if T != self.action_horizon:
            raise ValueError(
                f"Expected noised_actions.shape[1] == action_horizon={self.action_horizon}, "
                f"but got {T}."
            )

        if action_queries.shape[:2] != noised_actions.shape[:2]:
            raise ValueError(
                f"Expected action_queries and noised_actions to share [B, T], "
                f"but got {tuple(action_queries.shape[:2])} and {tuple(noised_actions.shape[:2])}."
            )

        if timesteps.dim() != 1 or timesteps.shape[0] != B:
            raise ValueError(
                f"Expected timesteps to have shape (B,), but got {tuple(timesteps.shape)}."
            )

        query_tokens = self.query_proj(action_queries.to(dtype=dtype))
        action_tokens = self.action_proj(noised_actions)

        pos_ids = torch.arange(T, dtype=torch.long, device=device)
        step_pos = self.step_pos(pos_ids).unsqueeze(0).to(dtype=dtype)
        query_tokens = query_tokens + step_pos
        action_tokens = action_tokens + step_pos

        time_token = self.time_encoder(timesteps[:, None]).to(device=device, dtype=dtype)
        time_token = self.time_proj(time_token)

        time_type = self.type_embedding.weight[0].view(1, 1, -1).to(dtype=dtype)
        query_type = self.type_embedding.weight[1].view(1, 1, -1).to(dtype=dtype)
        action_type = self.type_embedding.weight[2].view(1, 1, -1).to(dtype=dtype)

        time_token = time_token + time_type
        token_prefix = [time_token]

        if states is not None and self.state_proj is not None:
            if states.ndim == 3:
                states = states[:, -1, :]
            state_token = self.state_proj(
                states.to(device=device, dtype=dtype)
            ).unsqueeze(1)
            state_type = self.type_embedding.weight[1].view(1, 1, -1).to(dtype=dtype)
            state_token = state_token + state_type
            token_prefix.append(state_token)

        query_type = self.type_embedding.weight[2].view(1, 1, -1).to(dtype=dtype)
        action_type = self.type_embedding.weight[3].view(1, 1, -1).to(dtype=dtype)
        query_tokens = query_tokens + query_type
        action_tokens = action_tokens + action_type

        paired_tokens = torch.stack(
            [query_tokens, action_tokens],
            dim=2,
        )  # (B, T, 2, hidden_dim)
        interleaved_tokens = paired_tokens.reshape(B, 2 * T, self.hidden_dim)
        prefix_len = sum(token.shape[1] for token in token_prefix)
        tokens = torch.cat(token_prefix + [interleaved_tokens], dim=1)

        if self.use_attention_mask:
            attn_mask = self._build_attention_mask(
                prefix_len=prefix_len,
                action_horizon=T,
                device=device,
            )
            tokens = self.transformer(tokens, mask=attn_mask)
        else:
            tokens = self.transformer(tokens)
        interleaved_out = tokens[:, prefix_len:]

        query_out = interleaved_out[:, 0::2, :]
        action_out = interleaved_out[:, 1::2, :]

        pred_velocity = self.velocity_head(action_out)
        pred_query_actions = self.query_action_head(query_out)
        return pred_velocity, pred_query_actions


class FlowmatchingActionHead(nn.Module):
    """OFT-conditioned interleaved-token Flow Matching Action Head."""

    def __init__(
        self,
        action_model_type,
        action_hidden_dim,
        action_dim,
        future_action_window_size,
        state_dim=0,
        use_attention_mask=False,
    ):
        super().__init__()

        if action_model_type not in ["DiT-B", "DiT-L", "Interleaved-FM"]:
            raise ValueError(
                "FlowmatchingActionHead supports ['DiT-B', 'DiT-L', 'Interleaved-FM'] "
                f"as flow model aliases, but got {action_model_type}."
            )

        self.action_model_type = action_model_type
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(future_action_window_size) + 1

        self.num_inference_timesteps = 10
        self.noise_beta_alpha = 1.5
        self.noise_beta_beta = 1.0
        self.noise_s = 0.999
        self.noise_t_min = 0.001
        self.num_timestep_buckets = 1000

        # clean-action supervision for query tokens.
        self.query_action_loss_weight = 0.0
        self.use_attention_mask = bool(use_attention_mask)

        if action_model_type == "DiT-L":
            hidden_dim = 1536
            num_heads = 12
            num_layers = 6
        else:
            hidden_dim = 768
            num_heads = 12
            num_layers = 4

        self.flow_decoder = InterleavedFlowDecoder(
            action_hidden_dim=self.action_hidden_dim,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.0,
            use_attention_mask=self.use_attention_mask,
        )

        self.beta_dist = Beta(self.noise_beta_alpha, self.noise_beta_beta)

        self.state_dim = int(state_dim) if state_dim is not None else 0

    def sample_time(self, batch_size, device, dtype=torch.float32):
        # PI0-style convention:
        #   t close to 1 -> close to noise
        #   t close to 0 -> close to data/action
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return sample * self.noise_s + self.noise_t_min

    def _get_action_query_tokens(self, actions_hidden_states):
        if actions_hidden_states.shape[1] < self.action_horizon:
            raise ValueError(
                f"Expected at least {self.action_horizon} action query tokens, "
                f"but got sequence length {actions_hidden_states.shape[1]}."
            )
        return actions_hidden_states[:, -self.action_horizon:, :]

    def forward(
        self,
        actions_hidden_states=None,
        actions=None,
        states=None,
        **kwargs,
    ):
        """
        Training forward.

        Args:
            actions_hidden_states:
                Shape: (B, seq_len, action_hidden_dim). The last action_horizon
                tokens are assumed to be OFT latent action queries.

            actions:
                Ground-truth action chunk. Shape: (B, action_horizon, action_dim)

        Returns:
            Scalar training loss = flow velocity MSE + query clean-action loss.
        """

        if actions_hidden_states is None:
            raise ValueError(
                "FlowmatchingActionHead requires `actions_hidden_states` or `vl_embs`."
            )

        if actions is None:
            raise ValueError(
                "FlowmatchingActionHead training requires ground-truth `actions`."
            )

        if actions.shape[1] != self.action_horizon:
            raise ValueError(
                f"Expected actions.shape[1] == action_horizon={self.action_horizon}, "
                f"but got actions.shape[1]={actions.shape[1]}."
            )

        action_queries = self._get_action_query_tokens(actions_hidden_states).float()
        actions = actions.float()
        if states is not None:
            states = states.float()

        noise = torch.randn(
            actions.shape,
            device=actions.device,
            dtype=torch.float32,
        )

        t = self.sample_time(
            batch_size=actions.shape[0],
            device=actions.device,
            dtype=torch.float32,
        )
        t_expanded = t[:, None, None]

        noised_actions = t_expanded * noise + (1.0 - t_expanded) * actions
        target_velocity = noise - actions

        t_discretized = (t * self.num_timestep_buckets).long().clamp(
            min=0,
            max=self.num_timestep_buckets - 1,
        )

        pred_velocity, pred_query_actions = self.flow_decoder(
            action_queries=action_queries,
            noised_actions=noised_actions,
            timesteps=t_discretized,
            states=states,
        )

        flow_loss = F.mse_loss(pred_velocity.float(), target_velocity.float())
        query_action_loss = F.l1_loss(pred_query_actions.float(), actions.float())
        loss = flow_loss + self.query_action_loss_weight * query_action_loss
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        actions_hidden_states=None,
        states=None,
        **kwargs,
    ):
        """
        Inference.

        Returns:
            actions:
                Shape: (B, action_horizon, action_dim)
        """

        if actions_hidden_states is None:
            raise ValueError(
                "FlowmatchingActionHead.predict_action requires `actions_hidden_states` or `vl_embs`."
            )

        action_queries = self._get_action_query_tokens(actions_hidden_states).float()
        if states is not None:
            states = states.float()

        batch_size = action_queries.shape[0]
        device = action_queries.device

        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = -1.0 / float(num_steps)

        for step in range(num_steps):
            t_cont = 1.0 - step / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            t_discretized = max(0, min(self.num_timestep_buckets - 1, t_discretized))

            timesteps = torch.full(
                size=(batch_size,),
                fill_value=t_discretized,
                dtype=torch.long,
                device=device,
            )

            pred_velocity, _ = self.flow_decoder(
                action_queries=action_queries,
                noised_actions=actions,
                timesteps=timesteps,
                states=states,
            )

            actions = actions + dt * pred_velocity.float()

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# ============================================================
# Factory
# ============================================================

def get_action_model(config=None):
    """
    Factory: build ActionModel from global framework config.

    Supported action_model_type:
        - "MLP"            : original MLP regression action head
        - "Transformer"    : transformer regression action head
        - "DiT-B"          : interleaved flow decoder with B-sized hidden width
        - "DiT-L"          : interleaved flow decoder with L-sized hidden width
        - "Interleaved-FM" : explicit alias for the B-sized interleaved flow decoder

    Shared existing config fields:
        - action_hidden_dim
        - action_dim
        - future_action_window_size
    """
    action_model_cfg = config.framework.action_model

    model_type = str(action_model_cfg.action_model_type).strip()
    action_hidden_dim = int(action_model_cfg.action_hidden_dim)
    action_dim = int(action_model_cfg.action_dim)
    state_dim = int(getattr(action_model_cfg, "state_dim", 0))
    future_action_window_size = int(action_model_cfg.future_action_window_size)
    use_attention_mask = bool(getattr(action_model_cfg, "use_attention_mask", False))
    transformer_hidden_dim = int(getattr(action_model_cfg, "transformer_hidden_dim", action_hidden_dim))
    transformer_num_layers = int(getattr(action_model_cfg, "transformer_num_layers", 4))
    transformer_num_heads = int(getattr(action_model_cfg, "transformer_num_heads", 16))
    transformer_ffn_multiplier = int(getattr(action_model_cfg, "transformer_ffn_multiplier", 4))
    transformer_dropout = float(getattr(action_model_cfg, "transformer_dropout", 0.0))
    condition_dim = int(getattr(action_model_cfg, "condition_dim", 0))

    if model_type == "MLP":
        action_model = L1RegressionActionHead(
            input_dim=action_hidden_dim,
            hidden_dim=action_hidden_dim * 2,
            action_dim=action_dim,
            NUM_ACTIONS_CHUNK=1 + future_action_window_size,
        )

        return action_model

    if model_type == "Transformer":
        action_model = TransformerRegressionActionHead(
            input_dim=action_hidden_dim,
            hidden_dim=transformer_hidden_dim,
            action_dim=action_dim,
            NUM_ACTIONS_CHUNK=1 + future_action_window_size,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            ffn_multiplier=transformer_ffn_multiplier,
            dropout=transformer_dropout,
            state_dim=state_dim,
            condition_dim=condition_dim,
        )

        return action_model

    if model_type in ["DiT-B", "DiT-L", "Interleaved-FM"]:
        action_model = FlowmatchingActionHead(
            action_model_type=model_type,
            action_hidden_dim=action_hidden_dim,
            action_dim=action_dim,
            future_action_window_size=future_action_window_size,
            state_dim=state_dim,
            use_attention_mask=use_attention_mask,
        )

        return action_model

    raise ValueError(
        f"Unsupported action_model_type: {model_type}. "
        f"Expected one of ['MLP', 'Transformer', 'DiT-B', 'DiT-L', 'Interleaved-FM']."
    )
