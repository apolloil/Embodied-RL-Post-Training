"""
core/brc_components.py — Shared BRC architecture components.

Components:
  - BRCTaskEmbedding: 32-dim L2-normalized task embedding (BRC original)
  - BRCActorBackbone: BroNet backbone without final output layer
  - BRCTanhPolicy: BroNet actor with SquashedNormal output
  - BRCRewardNormalizer: Reward normalizer with BRC-faithful bootstrap
  - StateEncoder: MLP encoder for state-based observations (39 → 512)
  - _orthogonal_init: Weight initialization helper

Training hyperparameters live in experiments/state/shared_config.py.
Only Bro-specific architecture constants are defined here.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.networks import BronetBlock, SquashedNormal
from core.normalizer import RewardNormalizer
from experiments.state.shared_config import (
    STATE_DIM, ACT_DIM, NUM_BINS,
    CRITIC_WIDTH, CRITIC_DEPTH, ENSEMBLE_SIZE,
    ACTOR_LR, CRITIC_LR, TEMP_LR, TEMP_ADAM_B1,
)

# =====================================================================
# Bro-specific Constants (not shared with TF scripts)
# =====================================================================

TASK_EMB_DIM    = 32            # BRC uses 32-dim L2-normalized (TF: 512)
ACTOR_WIDTH     = 256           # BRC actor hidden width
ACTOR_DEPTH     = 1             # BRC actor BroNet depth (1 residual block)
LOG_STD_MIN     = -10.0
LOG_STD_MAX     = 2.0


# =====================================================================
# Weight Initialization
# =====================================================================

def _orthogonal_init(module, gain=math.sqrt(2)):
    """Apply orthogonal initialization matching BRC's default_init(sqrt(2))."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# =====================================================================
# BRC Task Embedding (32-dim, L2-normalized)
# =====================================================================

class BRCTaskEmbedding(nn.Module):
    """Task embedding with L2 normalization, faithful to BRC.

    BRC: 32-dim embedding, output = embed / ||embed||_2.
    Our TaskEmbedding: 512-dim, L2 normalized.
    """

    def __init__(self, num_tasks, embed_dim=TASK_EMB_DIM):
        super().__init__()
        self.embed = nn.Embedding(num_tasks, embed_dim)

    def forward(self, task_ids):
        emb = self.embed(task_ids)
        return F.normalize(emb, dim=-1)


# =====================================================================
# BRC Actor Backbone (BroNet without final output layer)
# =====================================================================

class BRCActorBackbone(nn.Module):
    """BroNet backbone *without* final output layer, for BRC actor.

    Architecture: Dense(input -> hidden) -> LN -> ReLU -> [BronetBlock] x depth
    Output dimension = hidden_dim (features, no projection).
    """

    def __init__(self, input_dim, hidden_dim=ACTOR_WIDTH, depth=ACTOR_DEPTH):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [BronetBlock(hidden_dim) for _ in range(depth)]
        )
        self.apply(lambda m: _orthogonal_init(m, gain=math.sqrt(2)))

    def forward(self, x):
        x = F.relu(self.ln_in(self.fc_in(x)))
        for block in self.blocks:
            x = block(x)
        return x


# =====================================================================
# BRC Tanh Policy (SquashedNormal)
# =====================================================================

class BRCTanhPolicy(nn.Module):
    """BRC actor: BroNet backbone → mean + log_std → SquashedNormal.

    log_std is parameterized via tanh squashing into [LOG_STD_MIN, LOG_STD_MAX].
    """

    def __init__(self, input_dim, action_dim=ACT_DIM,
                 hidden_dim=ACTOR_WIDTH, depth=ACTOR_DEPTH):
        super().__init__()
        self.backbone = BRCActorBackbone(input_dim, hidden_dim, depth)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        _orthogonal_init(self.mean_head, gain=math.sqrt(2))
        _orthogonal_init(self.log_std_head, gain=1.0)

    def forward(self, obs, temperature=1.0):
        features = self.backbone(obs)
        means = self.mean_head(features)
        raw_log_stds = self.log_std_head(features)
        log_stds = LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) * 0.5 * (
            1.0 + torch.tanh(raw_log_stds))
        stds = torch.exp(log_stds) * temperature
        return SquashedNormal(means, stds)


# =====================================================================
# BRC Reward Normalizer (override bootstrap logic)
# =====================================================================

class BRCRewardNormalizer(RewardNormalizer):
    """Reward normalizer with BRC-faithful bootstrap.

    BRC bootstrap: mean(rewards_in_trajectory) * effective_horizon
    (for truncated episodes, which is always the case in MetaWorld).

    Our base normalizer uses a fixed bootstrap of 20.0 * horizon.
    """

    def _calculate_return_stats(self, rewards):
        rewards = np.array(rewards, dtype=np.float32)
        T = len(rewards)
        if T == 0:
            return np.inf, -np.inf

        # BRC original: always bootstrap (MetaWorld terminal is always False)
        bootstrap = float(rewards.mean()) * self.effective_horizon
        values = np.zeros_like(rewards)
        running = bootstrap
        for t in reversed(range(T)):
            running = rewards[t] + self.discount * running
            values[t] = running
        return float(values.min()), float(values.max())


# =====================================================================
# State Encoder (MLP replacement for VisionEncoder)
# =====================================================================

class StateEncoder(nn.Module):
    """Simple MLP encoder: state_dim -> hidden -> hidden -> output_size.

    Replaces VisionEncoder for state-based observation input.
    Uses ReLU activations and LayerNorm for stable training.
    """

    def __init__(self, state_dim, output_size, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x):
        return self.net(x)
