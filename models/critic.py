"""
Centralised Critic network for AMAPPO.

Architecture:
    Input: global_state = concatenation of all agents' observations
           For M=4 agents, each obs=37 → global_state = 148-dim
    → Linear(148 → 128) → ReLU
    → GRU(input=128, hidden=64)
    → Linear(64 → 1)
    → Output: scalar V(s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int = 37,
        n_agents: int = 4,
        gru_hidden: int = 64,
    ):
        super().__init__()
        self.input_dim = obs_dim * n_agents  # 148
        self.gru_hidden = gru_hidden

        self.fc_in = nn.Linear(self.input_dim, 128)
        self.gru = nn.GRU(input_size=128, hidden_size=gru_hidden, batch_first=True)
        self.value_head = nn.Linear(gru_hidden, 1)

    def forward(
        self,
        global_obs: torch.Tensor,              # (input_dim,) or (B, input_dim)
        h_prev: Optional[torch.Tensor] = None, # (1, 1, 64) or (1, B, 64)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            value  : (1,)        — scalar state value
            h_next : (1, 1, 64) — updated GRU hidden state
        """
        unbatched = global_obs.dim() == 1
        if unbatched:
            global_obs = global_obs.unsqueeze(0)  # (1, input_dim)

        x = F.relu(self.fc_in(global_obs))  # (B, 128)
        x = x.unsqueeze(1)                  # (B, 1, 128) — single time-step

        gru_out, h_next = self.gru(x, h_prev)  # gru_out: (B, 1, 64)
        feat = gru_out.squeeze(1)              # (B, 64)

        value = self.value_head(feat)          # (B, 1)

        if unbatched:
            value = value.squeeze(0)  # (1,)
            # h_next remains (1, 1, 64)

        return value, h_next
