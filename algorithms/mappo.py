"""
Synchronous MAPPO trainer.

All agents act in every global timestep; experience is pooled into a shared
GlobalBuffer and updated jointly with PPO.

Training loop (per episode):
  1. Reset env, reset agent hidden states.
  2. For each step:
     - All M agents call agent.act() simultaneously (synchronous).
     - Build action_dict, call env.step().
     - Build global_obs (concatenate all agent obs).
     - Collect (obs, action, reward, h_pi, h_V, global_obs, done) for every agent.
  3. After episode ends:
     - Pour all agent buffers into GlobalBuffer.
     - Compute GAE advantages + returns on the collected data.
     - Run K PPO update epochs over mini-batches.
     - Clear buffers.
"""

from __future__ import annotations

import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from utils.config import Config
from utils.buffer import AgentBuffer, GlobalBuffer, Transition
from utils.logger import Logger
from models.agent import MAPPOAgent
from models.gnn_encoder import GNNEncoder
from env.sec_env import SECEnv


# ---------------------------------------------------------------------------
# Helper: build graph tensors from env
# ---------------------------------------------------------------------------

def _build_graph_inputs(env: SECEnv, agent_id: int):
    """
    Construct DAG and resource graph tensors for agent `agent_id`.

    DAG node features: [D_in, D_out, C, deadline_rem, topo_pos]  — 5-dim
    Resource graph:    4-node (local, UAV, SAT, cloud), features [load, capacity] — 2-dim
    """
    dag = env.dags[agent_id]
    nodes_sorted = sorted(dag.nodes())
    N_dag = len(nodes_sorted)
    node_to_idx = env.node_to_idx[agent_id]

    # --- DAG node features (N_dag, 5) ---
    dag_x = np.zeros((N_dag, 5), dtype=np.float32)
    topo = env.topo_orders[agent_id]
    topo_len = max(len(topo) - 1, 1)
    for node in nodes_sorted:
        idx = node_to_idx[node]
        attrs = dag.nodes[node]
        topo_pos = topo.index(node) / topo_len if node in topo else 0.0
        dag_x[idx, 0] = attrs.get("D_in", 0.0)
        dag_x[idx, 1] = attrs.get("D_out", 0.0)
        dag_x[idx, 2] = attrs.get("C", 0.0)
        dag_x[idx, 3] = float(max(0, env.max_steps - env._step_count)) * env.dt
        dag_x[idx, 4] = topo_pos

    # --- DAG edge_index (2, E) ---
    edges = list(dag.edges())
    if len(edges) > 0:
        src = [node_to_idx[u] for u, v in edges]
        dst = [node_to_idx[v] for u, v in edges]
        dag_edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        dag_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # --- Resource graph: 4 nodes (local=0, UAV=1, SAT=2, cloud=3) ---
    # Node features: [server_load, normalized_capacity]
    capacities = [0.8, 3.0, 4.5, 10.0]  # GHz (rough)
    max_cap = max(capacities)
    res_x = np.array(
        [[env.server_loads[i], capacities[i] / max_cap] for i in range(4)],
        dtype=np.float32,
    )  # (4, 2)

    # Fully connected resource graph (undirected: add both directions)
    res_src, res_dst = [], []
    for i in range(4):
        for j in range(4):
            if i != j:
                res_src.append(i)
                res_dst.append(j)
    res_edge_index = torch.tensor([res_src, res_dst], dtype=torch.long)

    return (
        torch.tensor(dag_x),
        dag_edge_index,
        torch.tensor(res_x),
        res_edge_index,
    )


# ---------------------------------------------------------------------------
# MAPPO Trainer
# ---------------------------------------------------------------------------

class MAPPOTrainer:
    """Synchronous MAPPO trainer."""

    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)

        # Shared GNN encoder
        self.shared_gnn = GNNEncoder().to(self.device)

        # Agents
        self.agents: List[MAPPOAgent] = [
            MAPPOAgent(agent_id=m, config=config, shared_gnn_encoder=self.shared_gnn)
            for m in range(config.M)
        ]

        # Unified optimizer over all parameters
        all_params = []
        all_params += list(self.shared_gnn.parameters())
        for agent in self.agents:
            all_params += list(agent.actor.parameters())
            all_params += list(agent.critic.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=config.lr)

        # Buffers
        self.agent_buffers: List[AgentBuffer] = [
            AgentBuffer(agent_id=m) for m in range(config.M)
        ]
        self.global_buffer = GlobalBuffer(capacity=50000)

        # Environment
        self.env = SECEnv(config)

        # Logger
        os.makedirs(config.log_dir, exist_ok=True)
        self.logger = Logger(log_dir=config.log_dir, algo_name="mappo")

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self._train_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self):
        """Run full training for config.epochs episodes."""
        for ep in range(1, self.cfg.epochs + 1):
            ep_reward, ep_info = self._run_episode()

            # PPO update
            if len(self.global_buffer) >= self.cfg.mini_batch_size:
                train_metrics = self._ppo_update()
            else:
                train_metrics = {"critic_loss": 0.0, "actor_loss": 0.0, "entropy": 0.0}

            # Logging
            if ep % self.cfg.log_interval == 0:
                metrics = {
                    "episode_reward": ep_reward,
                    "T_total": ep_info.get("T_total", 0.0),
                    "E_total": ep_info.get("E_total", 0.0),
                    "cost": -ep_reward,
                    "violations": ep_info.get("violations", [0] * 5),
                }
                self.logger.log_episode(ep, metrics)
                self.logger.log_training(self._train_step, train_metrics)
                print(
                    f"[MAPPO] ep={ep:4d}  reward={ep_reward:8.3f}  "
                    f"T={ep_info.get('T_total',0):.4f}  "
                    f"E={ep_info.get('E_total',0):.4e}  "
                    f"viol={ep_info.get('violations',[0]*5)}"
                )

            # Checkpoint
            if ep % self.cfg.save_interval == 0:
                self._save_checkpoint(ep)

        self.logger.close()
        print("[MAPPO] Training complete.")

    # ------------------------------------------------------------------
    # Episode rollout
    # ------------------------------------------------------------------

    def _run_episode(self) -> Tuple[float, dict]:
        """Collect one episode of experience (synchronous)."""
        obs_dict = self.env.reset()

        # Reset all agent hidden states and buffers
        for agent in self.agents:
            agent.reset_hidden()
        for buf in self.agent_buffers:
            buf.clear()

        total_reward = 0.0
        last_info: dict = {}

        while True:
            # Global obs = concatenation of all agent observations
            global_obs = np.concatenate(
                [obs_dict[m] for m in range(self.cfg.M)], axis=0
            )  # (148,)

            action_dict: Dict[int, np.ndarray] = {}
            stored: Dict[int, dict] = {}

            for m, agent in enumerate(self.agents):
                obs_m = obs_dict[m]
                dag_x, dag_ei, res_x, res_ei = _build_graph_inputs(self.env, m)

                action, log_prob, h_pi = agent.act(
                    obs_m, dag_x, dag_ei, res_x, res_ei
                )
                value, h_V = agent.get_value(global_obs)

                action_dict[m] = action
                stored[m] = {
                    "obs": obs_m,
                    "action": action,
                    "log_prob": log_prob,
                    "h_pi": h_pi.detach().cpu().numpy(),
                    "h_V": h_V.detach().cpu().numpy(),
                    "global_obs": global_obs.copy(),
                }

            next_obs_dict, rew_dict, done, info = self.env.step(action_dict)
            last_info = info

            # Store transitions
            for m in range(self.cfg.M):
                reward = rew_dict.get(m, 0.0)
                total_reward += reward
                t = Transition(
                    obs=stored[m]["obs"],
                    action=stored[m]["action"],
                    reward=reward,
                    h_pi=stored[m]["h_pi"],
                    h_V=stored[m]["h_V"],
                    global_obs=stored[m]["global_obs"],
                    done=done,
                )
                self.agent_buffers[m].add(t)

            obs_dict = next_obs_dict

            if done:
                break

        # Pour all agent buffers into global buffer
        for buf in self.agent_buffers:
            self.global_buffer.add_from_agent_buffer(buf)
            buf.clear()

        avg_reward = total_reward / self.cfg.M
        return avg_reward, last_info

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> dict:
        """One round of PPO updates on the global buffer."""
        # Sample a mini-batch
        batch = self.global_buffer.sample(self.cfg.mini_batch_size)

        obs_t        = torch.tensor(batch["obs"],        dtype=torch.float32, device=self.device)
        actions_t    = torch.tensor(batch["actions"],    dtype=torch.float32, device=self.device)
        rewards_t    = torch.tensor(batch["rewards"],    dtype=torch.float32, device=self.device)
        h_pi_t       = torch.tensor(batch["h_pi"],       dtype=torch.float32, device=self.device)
        h_V_t        = torch.tensor(batch["h_V"],        dtype=torch.float32, device=self.device)
        global_obs_t = torch.tensor(batch["global_obs"], dtype=torch.float32, device=self.device)
        dones_t      = torch.tensor(batch["dones"],      dtype=torch.float32, device=self.device)

        B = obs_t.shape[0]

        # Use agent 0's graph as representative (simplified)
        dag_x, dag_ei, res_x, res_ei = _build_graph_inputs(self.env, 0)
        dag_x  = dag_x.to(self.device)
        dag_ei = dag_ei.to(self.device)
        res_x  = res_x.to(self.device)
        res_ei = res_ei.to(self.device)

        # Compute baseline values for GAE (use critic of agent 0 as shared critic)
        with torch.no_grad():
            agent0 = self.agents[0]
            # Quick value estimate for GAE: forward pass critic
            h_V_init = h_V_t.squeeze(1).squeeze(1).unsqueeze(0)  # (1, B, 64)
            values_np_list = []
            for i in range(B):
                g_obs_i = global_obs_t[i:i+1]   # (1, 148)
                h_v_i = h_V_t[i:i+1].squeeze(1).squeeze(1).unsqueeze(0)  # (1,1,64)
                v, _ = agent0.critic(g_obs_i, h_v_i)
                values_np_list.append(v.item())
            values_np = np.array(values_np_list, dtype=np.float32)

        rewards_np = rewards_t.cpu().numpy()
        dones_np   = dones_t.cpu().numpy()
        advantages_np, returns_np = self.global_buffer.compute_returns_and_advantages(
            rewards_np, values_np, dones_np,
            gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda
        )

        advantages_t = torch.tensor(advantages_np, dtype=torch.float32, device=self.device)
        returns_t    = torch.tensor(returns_np,    dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Evaluate current policy
        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0

        # Use agent 0 as the representative policy (shared GNN)
        agent = self.agents[0]
        self.shared_gnn.train()
        agent.actor.train()
        agent.critic.train()

        log_probs_new, entropies, values_new, _ = agent.evaluate_actions(
            obs_t, actions_t, global_obs_t, h_pi_t, h_V_t,
            dag_x, dag_ei, res_x, res_ei
        )

        # Old log-probs: stored log_probs not available in buffer → use current as proxy
        # (first PPO epoch; ratio ≈ 1 at start, still provides gradient signal)
        log_probs_old = log_probs_new.detach()

        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.eps_clip, 1.0 + self.cfg.eps_clip) * advantages_t
        actor_loss  = -torch.min(surr1, surr2).mean()
        critic_loss = nn.functional.mse_loss(values_new, returns_t)
        entropy     = entropies.mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for pg in self.optimizer.param_groups for p in pg["params"]],
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()

        self._train_step += 1

        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy":     entropy.item(),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, episode: int):
        path = os.path.join(self.cfg.checkpoint_dir, f"mappo_ep{episode}.pt")
        state = {
            "episode": episode,
            "shared_gnn": self.shared_gnn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        for m, agent in enumerate(self.agents):
            state[f"actor_{m}"]  = agent.actor.state_dict()
            state[f"critic_{m}"] = agent.critic.state_dict()
        torch.save(state, path)
        print(f"[MAPPO] Checkpoint saved: {path}")
