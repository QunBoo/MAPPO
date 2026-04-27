"""
MAPPOAgent: wraps GNNEncoder, Actor, and Critic into a single agent object.

Observation layout:
  obs[0:5]     = task features (5)
  obs[5:25]    = upstream decisions (20)
  obs[25:-8]   = fine-grained server states (R = M + K + 2)
  obs[-8:]     = prev action (8)
"""

import torch
import numpy as np
from typing import Tuple, Optional


class MAPPOAgent:
    """Single MARL agent with shared GNN + independent Actor/Critic."""

    def __init__(self, agent_id: int, config, shared_gnn_encoder=None):
        self.agent_id = agent_id
        self.device = torch.device(config.device)
        self.resource_node_count = config.resource_node_count

        if shared_gnn_encoder is not None:
            self.gnn = shared_gnn_encoder
        else:
            from models.gnn_encoder import GNNEncoder
            self.gnn = GNNEncoder().to(self.device)

        from models.actor import Actor
        from models.critic import Critic

        self.actor = Actor(server_dim=self.resource_node_count).to(self.device)
        self.critic = Critic(obs_dim=config.obs_dim, n_agents=config.M).to(self.device)

        self.h_pi: Optional[torch.Tensor] = None
        self.h_V: Optional[torch.Tensor] = None

    def reset_hidden(self):
        self.h_pi = None
        self.h_V = None

    def _slice_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        upstream_dec = obs[..., 5:25]
        server_embed = obs[..., 25:-8]
        prev_action = obs[..., -8:]
        return upstream_dec, server_embed, prev_action

    def act(
        self,
        obs: np.ndarray,
        dag_x: torch.Tensor,
        dag_edge_index: torch.Tensor,
        res_x: torch.Tensor,
        res_edge_index: torch.Tensor,
    ) -> Tuple[np.ndarray, float, torch.Tensor]:
        self.gnn.eval()
        self.actor.eval()

        with torch.no_grad():
            dag_x = dag_x.to(self.device)
            dag_edge_index = dag_edge_index.to(self.device)
            res_x = res_x.to(self.device)
            res_edge_index = res_edge_index.to(self.device)

            gnn_embed = self.gnn(dag_x, dag_edge_index, res_x, res_edge_index)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            upstream_dec, server_embed, prev_action = self._slice_obs(obs_t)

            offload_logits, cont_out, log_prob, h_next = self.actor(
                gnn_embed, upstream_dec, prev_action, server_embed, self.h_pi
            )
            self.h_pi = h_next
            action = torch.cat([offload_logits, cont_out], dim=-1)

        return action.cpu().numpy(), log_prob.item(), self.h_pi

    def get_value(self, global_obs: np.ndarray) -> Tuple[float, torch.Tensor]:
        self.critic.eval()

        with torch.no_grad():
            global_obs_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device)
            value, h_next = self.critic(global_obs_t, self.h_V)
            self.h_V = h_next

        return value.item(), self.h_V

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        global_obs_batch: torch.Tensor,
        h_pi_batch: torch.Tensor,
        h_V_batch: torch.Tensor,
        dag_x: torch.Tensor,
        dag_edge_index: torch.Tensor,
        res_x: torch.Tensor,
        res_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = obs_batch.shape[0]

        obs_batch = obs_batch.to(self.device)
        actions_batch = actions_batch.to(self.device)
        global_obs_batch = global_obs_batch.to(self.device)
        h_pi_batch = h_pi_batch.to(self.device)
        h_V_batch = h_V_batch.to(self.device)
        dag_x = dag_x.to(self.device)
        dag_edge_index = dag_edge_index.to(self.device)
        res_x = res_x.to(self.device)
        res_edge_index = res_edge_index.to(self.device)

        gnn_embed = self.gnn(dag_x, dag_edge_index, res_x, res_edge_index)
        gnn_embed_batch = gnn_embed.unsqueeze(0).expand(B, -1)

        upstream_dec, server_embed, prev_action = self._slice_obs(obs_batch)
        discrete_actions = actions_batch[:, :4].argmax(dim=-1)

        h_pi_init = h_pi_batch.squeeze(1).squeeze(1).unsqueeze(0)
        log_probs, entropies, h_pi_next = self.actor.evaluate(
            gnn_embed_batch,
            upstream_dec,
            prev_action,
            server_embed,
            h_pi_init,
            discrete_actions,
        )
        h_pi_new = h_pi_next.squeeze(0).unsqueeze(1).unsqueeze(1)

        h_V_init = h_V_batch.squeeze(1).squeeze(1).unsqueeze(0)
        values, _ = self.critic(global_obs_batch, h_V_init)
        values = values.squeeze(-1)

        return log_probs, entropies, values, h_pi_new

    def parameters(self):
        return (
            list(self.gnn.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
