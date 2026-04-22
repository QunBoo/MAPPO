"""
AMAPPOv2 Trainer — async MAPPO with paper-aligned encoder-decoder architecture.

Key differences from AMAPPOTrainer:
  - Uses GNNEncoderV2 (returns node-level embeddings)
  - Uses MAPPOAgentV2 (encode-once + step-level decode)
  - _run_episode() calls agent.encode() at episode start
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from utils.config import Config
from utils.buffer import AgentBuffer, GlobalBuffer, Transition
from utils.logger import Logger
from models.v2.gnn_encoder_v2 import GNNEncoderV2
from models.v2.agent_v2 import MAPPOAgentV2
from env.sec_env import SECEnv


def _build_graph_inputs_v2(env: SECEnv, agent_id: int):
    """
    Build DAG and resource graph tensors from environment.
    (Same logic as mappo._build_graph_inputs, independently implemented.)
    """
    import torch
    dag = env.dags[agent_id]
    nodes_sorted = sorted(dag.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_sorted)}

    features = []
    for n in nodes_sorted:
        d = dag.nodes[n]
        features.append([
            float(d.get("D_in", 0.0)),
            float(d.get("D_out", 0.0)),
            float(d.get("C", 0.0)),
            float(d.get("deadline_rem", 0.0)),
            float(d.get("topo_pos", 0.0)),
        ])
    dag_x = torch.tensor(features, dtype=torch.float32)

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in dag.edges()]
    if edges:
        dag_ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        dag_ei = torch.zeros((2, 0), dtype=torch.long)

    # Resource graph: 4 nodes
    res_features = []
    for srv in ["local", "uav", "sat", "cloud"]:
        info = env.resource_graph.get(srv, {})
        res_features.append([
            float(info.get("load", 0.0)),
            float(info.get("capacity", 1.0)),
        ])
    res_x = torch.tensor(res_features, dtype=torch.float32)

    res_nodes = 4
    res_edges = [[i, j] for i in range(res_nodes) for j in range(res_nodes) if i != j]
    res_ei = torch.tensor(res_edges, dtype=torch.long).t().contiguous()

    return dag_x, dag_ei, res_x, res_ei


class AMAPPOv2Trainer:
    """AMAPPOv2: node-level embeddings + GRU + attention async MAPPO trainer."""

    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)

        # Shared GNNEncoderV2
        self.shared_encoder = GNNEncoderV2().to(self.device)

        # agent_types from config, default all LEO if not present
        agent_types = getattr(config, "agent_types", ["LEO"] * config.M)

        self.agents: List[MAPPOAgentV2] = [
            MAPPOAgentV2(
                agent_id=m,
                agent_type=agent_types[m],
                config=config,
                shared_encoder=self.shared_encoder,
            )
            for m in range(config.M)
        ]

        # Unified optimizer
        all_params = list(self.shared_encoder.parameters())
        for agent in self.agents:
            all_params += list(agent.actor.parameters())
            all_params += list(agent.critic.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=config.lr)

        # Buffers
        self.agent_buffers: List[AgentBuffer] = [
            AgentBuffer(agent_id=m) for m in range(config.M)
        ]
        self.global_buffer = GlobalBuffer(capacity=50000)

        # Environment and logging
        self.env = SECEnv(config)
        os.makedirs(config.log_dir, exist_ok=True)
        self.logger = Logger(log_dir=config.log_dir, algo_name="amappo_v2")
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self._train_step = 0
        self._agent_decision_count: np.ndarray = np.zeros(config.M, dtype=int)

    # ------------------------------------------------------------------

    def train(self):
        for ep in range(1, self.cfg.epochs + 1):
            ep_reward, ep_info = self._run_episode()

            if len(self.global_buffer) >= self.cfg.mini_batch_size:
                train_metrics = self._ppo_update()
            else:
                train_metrics = {"critic_loss": 0.0, "actor_loss": 0.0, "entropy": 0.0}

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
                decisions_per_agent = self._agent_decision_count.tolist()
                print(
                    f"[AMAPPOv2] ep={ep:4d}  reward={ep_reward:8.3f}  "
                    f"actor_loss={train_metrics['actor_loss']:.4f}  "
                    f"critic_loss={train_metrics['critic_loss']:.4f}  "
                    f"entropy={train_metrics['entropy']:.4f}  "
                    f"T={ep_info.get('T_total', 0):.4f}  "
                    f"E={ep_info.get('E_total', 0):.4e}  "
                    f"decisions={decisions_per_agent}",
                    flush=True,
                )
                self._agent_decision_count[:] = 0

            if ep % self.cfg.save_interval == 0:
                self._save_checkpoint(ep)

        self.logger.close()
        print("[AMAPPOv2] Training complete.", flush=True)

    # ------------------------------------------------------------------

    def _run_episode(self) -> Tuple[float, dict]:
        obs_dict = self.env.reset()

        for agent in self.agents:
            agent.reset_episode()
        for buf in self.agent_buffers:
            buf.clear()

        # <-- Key difference from AMAPPOTrainer: encode once -->
        for m, agent in enumerate(self.agents):
            dag_x, dag_ei, res_x, res_ei = _build_graph_inputs_v2(self.env, m)
            agent.encode(dag_x, dag_ei, res_x, res_ei)

        local_clocks: np.ndarray = np.zeros(self.cfg.M, dtype=int)
        total_reward = 0.0
        last_info: dict = {}
        global_step = 0
        last_obs: Dict[int, np.ndarray] = {m: obs_dict[m] for m in range(self.cfg.M)}
        action_dict: Dict[int, np.ndarray] = {
            m: np.zeros(self.cfg.action_dim) for m in range(self.cfg.M)
        }

        def _global_obs() -> np.ndarray:
            return np.concatenate([last_obs[m] for m in range(self.cfg.M)], axis=0)

        while True:
            global_obs = _global_obs()
            available_agents = [m for m in range(self.cfg.M) if global_step >= local_clocks[m]]

            for m in available_agents:
                agent = self.agents[m]
                obs_m = last_obs[m]

                action, log_prob, h_pi = agent.act(obs_m)
                value, h_V = agent.get_value(global_obs)
                # Pad action to env ACTION_DIM (8) if actor output is shorter
                if action.shape[0] < self.cfg.action_dim:
                    action = np.concatenate([
                        action,
                        np.zeros(self.cfg.action_dim - action.shape[0], dtype=np.float32),
                    ])
                action_dict[m] = action

                task_node = self.env._current_task(m)
                if task_node is not None:
                    c_cycles = self.env.dags[m].nodes[task_node].get("C", 1.0)
                    mean_slots = max(1, int(c_cycles / 0.5))
                    exec_slots = max(1, np.random.geometric(1.0 / mean_slots))
                else:
                    exec_slots = 1

                local_clocks[m] = global_step + exec_slots
                self._agent_decision_count[m] += 1

                t = Transition(
                    obs=obs_m,
                    action=action,
                    reward=0.0,
                    h_pi=h_pi.detach().cpu().numpy(),
                    h_V=h_V.detach().cpu().numpy(),
                    global_obs=global_obs.copy(),
                    done=False,
                    log_prob=log_prob,
                )
                self.agent_buffers[m].add(t)

            next_obs_dict, rew_dict, done, info = self.env.step(action_dict)
            last_info = info

            for m in available_agents:
                reward = rew_dict.get(m, 0.0)
                total_reward += reward
                buf = self.agent_buffers[m]
                if len(buf.transitions) > 0:
                    buf.transitions[-1].reward = reward
                    buf.transitions[-1].done   = done

            for m in range(self.cfg.M):
                last_obs[m] = next_obs_dict[m]

            global_step += 1
            if done:
                break

        for buf in self.agent_buffers:
            self.global_buffer.add_from_agent_buffer(buf)
            buf.clear()

        avg_reward = total_reward / max(1, sum(self._agent_decision_count))
        return avg_reward, last_info

    # ------------------------------------------------------------------

    def _ppo_update(self) -> dict:
        batch = self.global_buffer.sample(self.cfg.mini_batch_size)

        obs_t        = torch.tensor(batch["obs"],        dtype=torch.float32, device=self.device)
        actions_t    = torch.tensor(batch["actions"],    dtype=torch.float32, device=self.device)
        rewards_t    = torch.tensor(batch["rewards"],    dtype=torch.float32, device=self.device)
        h_pi_t       = torch.tensor(batch["h_pi"],       dtype=torch.float32, device=self.device)
        h_V_t        = torch.tensor(batch["h_V"],        dtype=torch.float32, device=self.device)
        global_obs_t = torch.tensor(batch["global_obs"], dtype=torch.float32, device=self.device)
        dones_t      = torch.tensor(batch["dones"],      dtype=torch.float32, device=self.device)
        log_probs_old_t = torch.tensor(batch["log_probs"], dtype=torch.float32, device=self.device)

        B = obs_t.shape[0]
        dag_x, dag_ei, res_x, res_ei = _build_graph_inputs_v2(self.env, 0)
        dag_x  = dag_x.to(self.device)
        dag_ei = dag_ei.to(self.device)
        res_x  = res_x.to(self.device)
        res_ei = res_ei.to(self.device)

        with torch.no_grad():
            agent0 = self.agents[0]
            values_np_list = []
            for i in range(B):
                g_obs_i = global_obs_t[i:i+1]
                h_v_i   = h_V_t[i:i+1].squeeze(1).squeeze(1).unsqueeze(0)
                v, _ = agent0.critic(g_obs_i, h_v_i)
                values_np_list.append(v.item())
            values_np = np.array(values_np_list, dtype=np.float32)

        rewards_np = rewards_t.cpu().numpy()
        dones_np   = dones_t.cpu().numpy()
        advantages_np, returns_np = self.global_buffer.compute_returns_and_advantages(
            rewards_np, values_np, dones_np,
            gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda,
        )

        advantages_t = torch.tensor(advantages_np, dtype=torch.float32, device=self.device)
        returns_t    = torch.tensor(returns_np,    dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Aggregate loss over all agents (parameter sharing)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for agent in self.agents:
            self.shared_encoder.train()
            agent.actor.train()
            agent.critic.train()

            log_probs_new, entropies, values_new, _ = agent.evaluate_actions(
                obs_t, actions_t, global_obs_t, h_pi_t, h_V_t,
                dag_x, dag_ei, res_x, res_ei,
            )

            ratio  = torch.exp(log_probs_new - log_probs_old_t)
            surr1  = ratio * advantages_t
            surr2  = torch.clamp(ratio, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages_t
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

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()

        M = len(self.agents)
        return {
            "actor_loss":  total_actor_loss / M,
            "critic_loss": total_critic_loss / M,
            "entropy":     total_entropy / M,
        }

    # ------------------------------------------------------------------

    def _save_checkpoint(self, episode: int):
        path = os.path.join(self.cfg.checkpoint_dir, f"amappo_v2_ep{episode}.pt")
        state = {
            "episode": episode,
            "shared_encoder": self.shared_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        for m, agent in enumerate(self.agents):
            state[f"actor_{m}"]  = agent.actor.state_dict()
            state[f"critic_{m}"] = agent.critic.state_dict()
        torch.save(state, path)
        print(f"[AMAPPOv2] Checkpoint saved: {path}", flush=True)
