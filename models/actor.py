"""
Actor network for AMAPPO.

Architecture:
    Input: gnn_embedding(128) + upstream_decisions(20) + prev_action(8) + server_embedding(R)
    -> GRU(input=156+R, hidden=64)
    -> Discrete head:   Linear(64 -> 4) -> offload logits
    -> Continuous head: Linear(64 -> 4) -> [bw_logit, comp_logit, dx, dy]
"""

import torch
import torch.nn as nn
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
        self.server_dim = server_dim
        self.input_dim = gnn_embed_dim + upstream_dim + prev_action_dim + server_dim
        self.gru_hidden = gru_hidden

        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=gru_hidden, batch_first=True)
        self.discrete_head = nn.Linear(gru_hidden, 4)
        self.continuous_head = nn.Linear(gru_hidden, 4)

    def _prepare_inputs(
        self,
        gnn_embed: torch.Tensor,
        upstream_dec: torch.Tensor,
        prev_action: torch.Tensor,
        server_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        unbatched = gnn_embed.dim() == 1
        if unbatched:
            gnn_embed = gnn_embed.unsqueeze(0)
            upstream_dec = upstream_dec.unsqueeze(0)
            prev_action = prev_action.unsqueeze(0)
            server_embed = server_embed.unsqueeze(0)

        x = torch.cat([gnn_embed, upstream_dec, prev_action, server_embed], dim=-1)
        return x.unsqueeze(1), unbatched

    def forward(
        self,
        gnn_embed: torch.Tensor,
        upstream_dec: torch.Tensor,
        prev_action: torch.Tensor,
        server_embed: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, unbatched = self._prepare_inputs(gnn_embed, upstream_dec, prev_action, server_embed)

        gru_out, h_next = self.gru(x, h_prev)
        feat = gru_out.squeeze(1)

        offload_logits = self.discrete_head(feat)
        raw_cont = self.continuous_head(feat)
        bw_comp = torch.sigmoid(raw_cont[..., :2])
        displacement = raw_cont[..., 2:]
        cont_out = torch.cat([bw_comp, displacement], dim=-1)

        dist = Categorical(logits=offload_logits)
        action_discrete = dist.sample()
        log_prob = dist.log_prob(action_discrete)

        if unbatched:
            offload_logits = offload_logits.squeeze(0)
            cont_out = cont_out.squeeze(0)
            log_prob = log_prob.squeeze(0)

        return offload_logits, cont_out, log_prob, h_next

    def evaluate(
        self,
        gnn_embed: torch.Tensor,
        upstream_dec: torch.Tensor,
        prev_action: torch.Tensor,
        server_embed: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, unbatched = self._prepare_inputs(gnn_embed, upstream_dec, prev_action, server_embed)

        gru_out, h_next = self.gru(x, h_prev)
        feat = gru_out.squeeze(1)
        offload_logits = self.discrete_head(feat)
        dist = Categorical(logits=offload_logits)

        if unbatched:
            action = action.unsqueeze(0) if action.dim() == 0 else action

        log_prob = dist.log_prob(action.long())
        entropy = dist.entropy()

        if unbatched:
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)

        return log_prob, entropy, h_next
