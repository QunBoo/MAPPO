"""
Centralised Critic network for AMAPPO.

Architecture:
    Input: global_state = concatenation of all agents' observations
           global_state_dim = obs_dim * n_agents
    -> Linear(global_state_dim -> 128) -> ReLU
    -> GRU(input=128, hidden=64)
    -> Linear(64 -> 1)
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
        self.input_dim = obs_dim * n_agents
        self.gru_hidden = gru_hidden

        self.fc_in = nn.Linear(self.input_dim, 128)
        self.gru = nn.GRU(input_size=128, hidden_size=gru_hidden, batch_first=True)
        self.value_head = nn.Linear(gru_hidden, 1)

    def forward(
        self,
        global_obs: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        unbatched = global_obs.dim() == 1
        if unbatched:
            global_obs = global_obs.unsqueeze(0)

        x = F.relu(self.fc_in(global_obs))
        x = x.unsqueeze(1)

        gru_out, h_next = self.gru(x, h_prev)
        feat = gru_out.squeeze(1)
        value = self.value_head(feat)

        if unbatched:
            value = value.squeeze(0)

        return value, h_next
