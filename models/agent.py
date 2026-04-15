"""
MAPPOAgent: wraps GNNEncoder, Actor, and Critic into a single agent object.

Each agent owns:
  - A reference to a GNNEncoder (optionally shared across agents)
  - Its own Actor and Critic networks
  - Independent GRU hidden states (h_pi for actor, h_V for critic)

Observation layout (37-dim):
  obs[0:5]   = task features (5)
  obs[5:25]  = upstream decisions (20)
  obs[25:29] = server states (4)
  obs[29:37] = prev action (8)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class MAPPOAgent:
    """
    Single MARL agent. Holds shared GNN encoder + independent Actor/Critic.
    Manages its own GRU hidden states (h_pi for actor, h_V for critic).
    """

    def __init__(self, agent_id: int, config, shared_gnn_encoder=None):
        """
        Args:
            agent_id: integer agent index
            config:   Config object with obs_dim, action_dim, gru_hidden, device
            shared_gnn_encoder: if provided, share GNN encoder across agents
        """
        self.agent_id = agent_id
        self.device = torch.device(config.device)

        # GNN encoder — shared if provided, otherwise create a private one
        if shared_gnn_encoder is not None:
            self.gnn = shared_gnn_encoder
        else:
            from models.gnn_encoder import GNNEncoder
            self.gnn = GNNEncoder().to(self.device)

        from models.actor import Actor
        from models.critic import Critic

        self.actor = Actor().to(self.device)
        self.critic = Critic(obs_dim=config.obs_dim, n_agents=config.M).to(self.device)

        # Independent GRU hidden states; None means "use zero-initialised state"
        self.h_pi: Optional[torch.Tensor] = None  # (1, 1, 64)
        self.h_V: Optional[torch.Tensor] = None   # (1, 1, 64)

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------

    def reset_hidden(self):
        """Reset GRU hidden states at episode start."""
        self.h_pi = None
        self.h_V = None

    # ------------------------------------------------------------------
    # Observation slicing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_obs(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split a (37,) or (B, 37) observation tensor into its components.

        Returns:
            upstream_dec : (..., 20)
            server_embed : (..., 4)
            prev_action  : (..., 8)
        """
        upstream_dec = obs[..., 5:25]    # 20-dim
        server_embed = obs[..., 25:29]   # 4-dim
        prev_action  = obs[..., 29:37]   # 8-dim
        return upstream_dec, server_embed, prev_action

    # ------------------------------------------------------------------
    # act — inference / data-collection
    # ------------------------------------------------------------------

    def act(
        self,
        obs: np.ndarray,
        dag_x: torch.Tensor,
        dag_edge_index: torch.Tensor,
        res_x: torch.Tensor,
        res_edge_index: torch.Tensor,
    ) -> Tuple[np.ndarray, float, torch.Tensor]:
        """
        Sample an action given a single observation.

        Args:
            obs:            (37,) numpy observation array
            dag_x:          (N, 5)   task DAG node features
            dag_edge_index: (2, E)   task DAG edges
            res_x:          (4, 2)   resource graph node features
            res_edge_index: (2, E_r) resource graph edges

        Returns:
            action:   (8,) numpy array = [offload_logits(4), cont_out(4)]
            log_prob: scalar float
            h_pi:     current actor hidden state (1, 1, 64) for buffer storage
        """
        self.gnn.eval()
        self.actor.eval()

        with torch.no_grad():
            # Move graph data to device
            dag_x          = dag_x.to(self.device)
            dag_edge_index = dag_edge_index.to(self.device)
            res_x          = res_x.to(self.device)
            res_edge_index = res_edge_index.to(self.device)

            # GNN embedding — (128,)
            gnn_embed = self.gnn(dag_x, dag_edge_index, res_x, res_edge_index)

            # Slice observation
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            upstream_dec, server_embed, prev_action = self._slice_obs(obs_t)

            # Actor forward pass (unbatched)
            offload_logits, cont_out, log_prob, h_next = self.actor(
                gnn_embed, upstream_dec, prev_action, server_embed, self.h_pi
            )

            # Update hidden state in-place
            self.h_pi = h_next  # (1, 1, 64)

            # Build 8-dim action vector
            action = torch.cat([offload_logits, cont_out], dim=-1)  # (8,)

        return action.cpu().numpy(), log_prob.item(), self.h_pi

    # ------------------------------------------------------------------
    # get_value — inference / data-collection
    # ------------------------------------------------------------------

    def get_value(
        self,
        global_obs: np.ndarray,
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute a state-value estimate from the global observation.

        Args:
            global_obs: (148,) concatenated observations of all agents

        Returns:
            value: scalar float
            h_V:   critic hidden state (1, 1, 64) for buffer storage
        """
        self.critic.eval()

        with torch.no_grad():
            global_obs_t = torch.as_tensor(
                global_obs, dtype=torch.float32, device=self.device
            )
            value, h_next = self.critic(global_obs_t, self.h_V)  # value: (1,)
            self.h_V = h_next  # (1, 1, 64)

        return value.item(), self.h_V

    # ------------------------------------------------------------------
    # evaluate_actions — PPO update
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,        # (B, 37)
        actions_batch: torch.Tensor,    # (B, 8)
        global_obs_batch: torch.Tensor, # (B, 148)
        h_pi_batch: torch.Tensor,       # (B, 1, 1, 64)
        h_V_batch: torch.Tensor,        # (B, 1, 1, 64)
        dag_x: torch.Tensor,            # (N, 5)  — representative graph for batch
        dag_edge_index: torch.Tensor,   # (2, E)
        res_x: torch.Tensor,            # (4, 2)
        res_edge_index: torch.Tensor,   # (2, E_r)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a batch of (obs, action) pairs for the PPO update.

        The GNN is run once with the provided graph inputs and the resulting
        128-dim embedding is broadcast across the batch (simplified approach
        consistent with using a representative graph state per mini-batch).

        The discrete action index used for evaluation is derived from
        actions_batch[:, :4] by taking the argmax of the stored logits.

        Args:
            obs_batch:        (B, 37)
            actions_batch:    (B, 8)   — [offload_logits(4), cont(4)]
            global_obs_batch: (B, 148)
            h_pi_batch:       (B, 1, 1, 64) — per-sample initial actor hidden
            h_V_batch:        (B, 1, 1, 64) — per-sample initial critic hidden
            dag_x, dag_edge_index, res_x, res_edge_index: representative graph

        Returns:
            log_probs:  (B,)
            entropies:  (B,)
            values:     (B,)
            h_pi_new:   (B, 1, 1, 64)
        """
        B = obs_batch.shape[0]

        # Move everything to device
        obs_batch        = obs_batch.to(self.device)
        actions_batch    = actions_batch.to(self.device)
        global_obs_batch = global_obs_batch.to(self.device)
        h_pi_batch       = h_pi_batch.to(self.device)   # (B, 1, 1, 64)
        h_V_batch        = h_V_batch.to(self.device)    # (B, 1, 1, 64)
        dag_x            = dag_x.to(self.device)
        dag_edge_index   = dag_edge_index.to(self.device)
        res_x            = res_x.to(self.device)
        res_edge_index   = res_edge_index.to(self.device)

        # --- GNN: run once, broadcast across batch ---
        gnn_embed = self.gnn(dag_x, dag_edge_index, res_x, res_edge_index)  # (128,)
        gnn_embed_batch = gnn_embed.unsqueeze(0).expand(B, -1)               # (B, 128)

        # --- Slice obs ---
        upstream_dec, server_embed, prev_action = self._slice_obs(obs_batch)
        # Each: (B, 20), (B, 4), (B, 8)

        # --- Recover discrete action indices from stored logits ---
        # actions_batch[:, :4] are the offload logits recorded at collection time
        offload_logit_stored = actions_batch[:, :4]             # (B, 4)
        discrete_actions = offload_logit_stored.argmax(dim=-1)  # (B,)

        # --- Actor evaluate (batched) ---
        # h_pi_batch is (B, 1, 1, 64); GRU expects h_prev as (num_layers, B, hidden)
        # Reshape: (B, 1, 1, 64) → (1, B, 64)
        h_pi_init = h_pi_batch.squeeze(1).squeeze(1).unsqueeze(0)  # (1, B, 64)

        log_probs, entropies, h_pi_next = self.actor.evaluate(
            gnn_embed_batch,   # (B, 128)
            upstream_dec,      # (B, 20)
            prev_action,       # (B, 8)
            server_embed,      # (B, 4)
            h_pi_init,         # (1, B, 64)
            discrete_actions,  # (B,)
        )
        # log_probs: (B,), entropies: (B,), h_pi_next: (1, B, 64)

        # Reshape h_pi_next back to (B, 1, 1, 64)
        h_pi_new = h_pi_next.squeeze(0).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 64)

        # --- Critic (batched) ---
        # h_V_batch: (B, 1, 1, 64) → (1, B, 64)
        h_V_init = h_V_batch.squeeze(1).squeeze(1).unsqueeze(0)  # (1, B, 64)

        values, _ = self.critic(global_obs_batch, h_V_init)  # values: (B, 1)
        values = values.squeeze(-1)                           # (B,)

        return log_probs, entropies, values, h_pi_new

    # ------------------------------------------------------------------
    # Parameters — for the optimizer
    # ------------------------------------------------------------------

    def parameters(self):
        """Return all trainable parameters (for optimizer)."""
        return (
            list(self.gnn.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
