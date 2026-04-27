import torch
import numpy as np
from typing import Optional, List, Dict, Tuple

from models.v2.gnn_encoder_v2 import GNNEncoderV2
from models.v2.actor_v2 import ActorV2
from models.critic import Critic


def _kahn_topo_sort(n_nodes: int, edge_index: torch.Tensor) -> List[int]:
    """
    Kahn's algorithm for topological sorting.
    Args:
        n_nodes:    total number of nodes
        edge_index: (2, E) directed edges [src, dst]
    Returns:
        topologically sorted list of node indices
    """
    in_degree = [0] * n_nodes
    adj: List[List[int]] = [[] for _ in range(n_nodes)]

    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)
            in_degree[d] += 1

    queue = [i for i in range(n_nodes) if in_degree[i] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != n_nodes:
        # Cycle detected (should not happen for DAG), fallback to sequential
        result = list(range(n_nodes))
    return result


class MAPPOAgentV2:
    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        config,
        shared_encoder: Optional[GNNEncoderV2] = None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.device = torch.device(config.device)

        self.encoder = shared_encoder if shared_encoder is not None else GNNEncoderV2().to(self.device)
        self.actor = ActorV2(agent_type=agent_type).to(self.device)
        self.critic = Critic(obs_dim=config.obs_dim, n_agents=config.M).to(self.device)

        # GRU hidden states
        self.h_pi: Optional[torch.Tensor] = None  # (1, 1, 64)
        self.h_V:  Optional[torch.Tensor] = None  # (1, 1, 64)

        # Episode-level cache
        self.node_embs:   Optional[torch.Tensor] = None  # (N, 64)
        self.server_embs: Optional[torch.Tensor] = None  # (R, 64)
        self.graph_enc:   Optional[torch.Tensor] = None  # (64,)
        self.topo_order:  Optional[List[int]] = None
        self.step_idx:    int = 0
        self.decisions:   Dict[int, torch.Tensor] = {}   # {task_id: action}

    # ------------------------------------------------------------------
    # encode: called once at episode start
    # ------------------------------------------------------------------

    def encode(
        self,
        dag_x: torch.Tensor,
        dag_edge_index: torch.Tensor,
        res_x: torch.Tensor,
        res_edge_index: torch.Tensor,
    ):
        self.encoder.eval()
        with torch.no_grad():
            dag_x          = dag_x.to(self.device)
            dag_edge_index = dag_edge_index.to(self.device)
            res_x          = res_x.to(self.device)
            res_edge_index = res_edge_index.to(self.device)

            node_embs, server_embs, graph_enc = self.encoder(
                dag_x, dag_edge_index, res_x, res_edge_index
            )

        self.node_embs   = node_embs
        self.server_embs = server_embs
        self.graph_enc   = graph_enc

        n_nodes = dag_x.shape[0]
        self.topo_order = _kahn_topo_sort(n_nodes, dag_edge_index.cpu())

        # Initialize GRU hidden state from graph_enc
        with torch.no_grad():
            self.h_pi = self.actor.init_hidden(graph_enc)  # (1, 1, 64)

        self.step_idx = 0
        self.decisions = {}

    # ------------------------------------------------------------------
    # act: called each step
    # ------------------------------------------------------------------

    def act(
        self,
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, float, torch.Tensor]:
        assert self.node_embs is not None, "Call encode() before act()"
        assert self.step_idx < len(self.topo_order), "step_idx out of range"

        self.actor.eval()
        with torch.no_grad():
            task_id = self.topo_order[self.step_idx]
            h_v_t = self.node_embs[task_id]                    # (64,)
            server_agg = self.server_embs.mean(dim=0)          # (64,)

            # Collect upstream decisions
            upstream_actions = [
                self.decisions[t]
                for t in self.topo_order[:self.step_idx]
                if t in self.decisions
            ]
            if upstream_actions:
                stacked = torch.stack(upstream_actions, dim=0)  # (K, d_a)
                # Pad or truncate to 64 dims for aggregation
                d = stacked.shape[-1]
                if d < 64:
                    pad = torch.zeros(*stacked.shape[:-1], 64 - d, device=self.device)
                    stacked = torch.cat([stacked, pad], dim=-1)
                else:
                    stacked = stacked[..., :64]
                L_us_agg = stacked.mean(dim=0)                 # (64,)
            else:
                L_us_agg = torch.zeros(64, device=self.device)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            a_prev = obs_t[-8:]                                # (8,)

            action, log_prob, h_next = self.actor(
                h_v_t, L_us_agg, server_agg, a_prev, self.h_pi, self.node_embs
            )

        self.decisions[task_id] = action.detach()
        self.step_idx += 1
        self.h_pi = h_next

        return action.cpu().numpy(), log_prob.item(), self.h_pi

    # ------------------------------------------------------------------
    # get_value: called each step (centralized Critic)
    # ------------------------------------------------------------------

    def get_value(self, global_obs: np.ndarray) -> Tuple[float, torch.Tensor]:
        self.critic.eval()
        with torch.no_grad():
            g_obs = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device)
            value, h_next = self.critic(g_obs, self.h_V)
            self.h_V = h_next
        return value.item(), self.h_V

    # ------------------------------------------------------------------
    # evaluate_actions: called during PPO update (batched)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs_batch:        torch.Tensor,   # (B, obs_dim)
        actions_batch:    torch.Tensor,   # (B, d_a)
        global_obs_batch: torch.Tensor,   # (B, obs_dim * M)
        h_pi_batch:       torch.Tensor,   # (B, 1, 1, 64)
        h_V_batch:        torch.Tensor,   # (B, 1, 1, 64)
        dag_x:            torch.Tensor,
        dag_edge_index:   torch.Tensor,
        res_x:            torch.Tensor,
        res_edge_index:   torch.Tensor,
        task_ids_batch:   torch.Tensor,   # (B,) task node index per sample
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = obs_batch.shape[0]
        obs_batch        = obs_batch.to(self.device)
        actions_batch    = actions_batch.to(self.device)
        global_obs_batch = global_obs_batch.to(self.device)
        h_pi_batch       = h_pi_batch.to(self.device)
        h_V_batch        = h_V_batch.to(self.device)

        # Encoder forward (re-encode during evaluation, training mode)
        dag_x          = dag_x.to(self.device)
        dag_edge_index = dag_edge_index.to(self.device)
        res_x          = res_x.to(self.device)
        res_edge_index = res_edge_index.to(self.device)

        self.encoder.train()
        node_embs, server_embs, _ = self.encoder(dag_x, dag_edge_index, res_x, res_edge_index)

        # Slice obs for a_prev and upstream decisions
        a_prev_batch   = obs_batch[:, -8:]            # (B, 8)
        upstream_batch = obs_batch[:, 5:25]           # (B, 20)

        # L_us_agg: pad upstream_batch to 64 dims
        pad = torch.zeros(B, 44, device=self.device)
        L_us_agg_batch = torch.cat([upstream_batch, pad], dim=-1)[:, :64]  # (B, 64)

        server_agg_batch = server_embs.mean(dim=0).unsqueeze(0).expand(B, -1)  # (B, 64)

        # h_v_t: index node_embs by the task_id each sample was generated from
        task_ids = task_ids_batch.long().to(self.device)  # (B,)
        h_v_t_batch = node_embs[task_ids]                 # (B, 64)

        h_pi_init = h_pi_batch.squeeze(1).squeeze(1).unsqueeze(0)  # (1, B, 64)

        discrete_actions = actions_batch[:, :self.actor.n_disc].argmax(dim=-1)  # (B,)
        continuous_actions = actions_batch[:, self.actor.n_disc:self.actor.n_disc + self.actor.n_cont]  # (B, n_cont)

        self.actor.train()
        log_probs, entropies, h_pi_next = self.actor.evaluate(
            h_v_t_batch, L_us_agg_batch, server_agg_batch,
            a_prev_batch, h_pi_init, node_embs, discrete_actions, continuous_actions,
        )

        h_V_init = h_V_batch.squeeze(1).squeeze(1).unsqueeze(0)  # (1, B, 64)
        self.critic.train()
        values, _ = self.critic(global_obs_batch, h_V_init)
        values = values.squeeze(-1)

        h_pi_new = h_pi_next.squeeze(0).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 64)
        return log_probs, entropies, values, h_pi_new

    # ------------------------------------------------------------------
    # reset_episode
    # ------------------------------------------------------------------

    def reset_episode(self):
        self.node_embs   = None
        self.server_embs = None
        self.graph_enc   = None
        self.topo_order  = None
        self.step_idx    = 0
        self.decisions   = {}
        self.h_pi        = None
        self.h_V         = None

    def parameters(self):
        return (
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
