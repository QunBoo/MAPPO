"""
Actor network for AMAPPO.

Architecture:
    Input: gnn_embedding(128) + upstream_decisions(20) + prev_action(8) + server_embedding(4)
         = 160-dim total input
    → GRU(input=160, hidden=64)
    → Discrete head:   Linear(64 → 4) → offload logits (Categorical)
    → Continuous head: Linear(64 → 4) → [bw_logit, comp_logit, dx, dy]
      - sigmoid applied to first 2 elements (bandwidth, compute allocation)
      - raw output for last 2 elements (UAV displacement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional


class Actor(nn.Module):
    def __init__(
        self,
        gnn_embed_dim: int = 128,
        upstream_dim: int = 20,
        prev_action_dim: int = 8,
        server_dim: int = 4,
        gru_hidden: int = 64,
    ):
        super().__init__()
        self.input_dim = gnn_embed_dim + upstream_dim + prev_action_dim + server_dim  # 160
        self.gru_hidden = gru_hidden

        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=gru_hidden, batch_first=True)

        # Discrete offload head: outputs logits for Categorical(4)
        self.discrete_head = nn.Linear(gru_hidden, 4)

        # Continuous head: [bw_logit, comp_logit, dx, dy]
        self.continuous_head = nn.Linear(gru_hidden, 4)

    def _prepare_inputs(
        self,
        gnn_embed: torch.Tensor,
        upstream_dec: torch.Tensor,
        prev_action: torch.Tensor,
        server_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Concatenate inputs and reshape to (B, 1, input_dim) for GRU.
        Returns the packed tensor and a flag indicating whether the input was unbatched.
        """
        unbatched = gnn_embed.dim() == 1
        if unbatched:
            gnn_embed = gnn_embed.unsqueeze(0)       # (1, 128)
            upstream_dec = upstream_dec.unsqueeze(0) # (1, 20)
            prev_action = prev_action.unsqueeze(0)   # (1, 8)
            server_embed = server_embed.unsqueeze(0) # (1, 4)

        x = torch.cat([gnn_embed, upstream_dec, prev_action, server_embed], dim=-1)  # (B, 160)
        x = x.unsqueeze(1)  # (B, 1, 160) — single time-step sequence
        return x, unbatched

    def forward(
        self,
        gnn_embed: torch.Tensor,       # (128,) or (B, 128)
        upstream_dec: torch.Tensor,    # (20,)  or (B, 20)
        prev_action: torch.Tensor,     # (8,)   or (B, 8)
        server_embed: torch.Tensor,    # (4,)   or (B, 4)
        h_prev: Optional[torch.Tensor] = None,  # (1, 1, 64) or (1, B, 64)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            offload_logits : (4,) or (B, 4)  — raw logits for Categorical distribution
            cont_out       : (4,) or (B, 4)  — [bw_sigmoid, comp_sigmoid, dx, dy]
            log_prob       : scalar ()        — log prob of sampled discrete action
            h_next         : (1, 1, 64)       — updated GRU hidden state
        """
        x, unbatched = self._prepare_inputs(gnn_embed, upstream_dec, prev_action, server_embed)
        # x: (B, 1, 160)

        gru_out, h_next = self.gru(x, h_prev)  # gru_out: (B, 1, 64), h_next: (1, B, 64)
        feat = gru_out.squeeze(1)              # (B, 64)

        # Discrete head
        offload_logits = self.discrete_head(feat)  # (B, 4)

        # Continuous head
        raw_cont = self.continuous_head(feat)       # (B, 4)
        bw_comp = torch.sigmoid(raw_cont[..., :2])  # (B, 2) — bandwidth, compute allocation
        displacement = raw_cont[..., 2:]            # (B, 2) — dx, dy, no activation
        cont_out = torch.cat([bw_comp, displacement], dim=-1)  # (B, 4)

        # Sample discrete action and compute log prob
        dist = Categorical(logits=offload_logits)
        action_discrete = dist.sample()             # (B,)
        log_prob = dist.log_prob(action_discrete)   # (B,)

        if unbatched:
            # Return single-sample (unbatched) tensors
            offload_logits = offload_logits.squeeze(0)  # (4,)
            cont_out = cont_out.squeeze(0)              # (4,)
            log_prob = log_prob.squeeze(0)              # scalar ()
            # h_next remains (1, 1, 64)

        return offload_logits, cont_out, log_prob, h_next

    def evaluate(
        self,
        gnn_embed: torch.Tensor,
        upstream_dec: torch.Tensor,
        prev_action: torch.Tensor,
        server_embed: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        action: torch.Tensor,  # discrete offload index, (B,) or scalar
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For PPO update: compute log_prob and entropy of given discrete action.

        Returns:
            log_prob   : (B,)  — log probability of supplied action
            entropy    : (B,)  — distribution entropy
            h_next     : (1, B, 64) — updated GRU hidden state
        """
        x, unbatched = self._prepare_inputs(gnn_embed, upstream_dec, prev_action, server_embed)

        gru_out, h_next = self.gru(x, h_prev)
        feat = gru_out.squeeze(1)  # (B, 64)

        offload_logits = self.discrete_head(feat)  # (B, 4)

        dist = Categorical(logits=offload_logits)

        if unbatched:
            action = action.unsqueeze(0) if action.dim() == 0 else action

        log_prob = dist.log_prob(action.long())  # (B,)
        entropy = dist.entropy()                 # (B,)

        if unbatched:
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)

        return log_prob, entropy, h_next
