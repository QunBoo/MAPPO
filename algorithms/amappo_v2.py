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

    res_x, res_ei = env.get_resource_graph_data()

    return dag_x, dag_ei, res_x, res_ei


class AMAPPOv2Trainer:
    """AMAPPOv2: node-level embeddings + GRU + attention async MAPPO trainer."""

    def __init__(self, config: Config):
        config.sync_derived_fields()
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
        self._dag_tensors: list = []   # per-agent (dag_x, dag_ei, res_x, res_ei), updated each episode

    # ------------------------------------------------------------------

    def train(self):
        train_metrics = {"critic_loss": 0.0, "actor_loss": 0.0, "entropy": 0.0}
        for ep in range(1, self.cfg.epochs + 1):
            ep_reward, ep_info = self._run_episode()

            # Fix C: accumulate update_every episodes, then PPO update + clear buffer
            if ep % self.cfg.update_every == 0:
                if len(self.global_buffer) >= self.cfg.mini_batch_size:
                    train_metrics = self._ppo_update()
                    self.global_buffer.clear()   # on-policy: discard stale data

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

            if ep % self.cfg.save_interval == 0:
                self._save_checkpoint(ep)

        self.logger.close()
        print("[AMAPPOv2] Training complete.", flush=True)

    # ------------------------------------------------------------------

    def _run_episode(self) -> Tuple[float, dict]:
        self._agent_decision_count[:] = 0   # reset per-episode counter at episode start
        obs_dict = self.env.reset()

        for agent in self.agents:
            agent.reset_episode()
        for buf in self.agent_buffers:
            buf.clear()

        # <-- Key difference from AMAPPOTrainer: encode once -->
        self._dag_tensors = []
        for m, agent in enumerate(self.agents):
            dag_x, dag_ei, res_x, res_ei = _build_graph_inputs_v2(self.env, m)
            agent.encode(dag_x, dag_ei, res_x, res_ei)
            self._dag_tensors.append((
                dag_x.to(self.device),
                dag_ei.to(self.device),
                res_x.to(self.device),
                res_ei.to(self.device),
            ))

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
                # Record the task_id used in act() so evaluate_actions can reproduce the same embedding
                current_task_id = agent.topo_order[agent.step_idx - 1]  # step_idx was incremented in act()
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
                    task_id=current_task_id,
                    agent_id=m,
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

        # Pre-compute GAE on each agent's sequential trajectory before shuffling into GlobalBuffer
        for m, buf in enumerate(self.agent_buffers):
            if len(buf.transitions) == 0:
                continue
            rewards_m = np.array([t.reward for t in buf.transitions], dtype=np.float32)
            dones_m   = np.array([float(t.done) for t in buf.transitions], dtype=np.float32)

            values_m = []
            self.agents[m].critic.eval()
            with torch.no_grad():
                for t in buf.transitions:
                    g_obs = torch.tensor(t.global_obs, dtype=torch.float32, device=self.device)
                    h_v_in = torch.tensor(t.h_V, dtype=torch.float32, device=self.device
                             ).squeeze(1).squeeze(1).unsqueeze(0)
                    v, _ = self.agents[m].critic(g_obs, h_v_in)
                    values_m.append(v.item())
            values_m = np.array(values_m, dtype=np.float32)

            adv_m, ret_m = self.global_buffer.compute_returns_and_advantages(
                rewards_m, values_m, dones_m,
                gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda,
            )
            for i, t in enumerate(buf.transitions):
                t.advantage = float(adv_m[i])
                t.ret       = float(ret_m[i])

        for buf in self.agent_buffers:
            self.global_buffer.add_from_agent_buffer(buf)
            buf.clear()

        ep_decisions = int(self._agent_decision_count.sum())
        avg_reward = total_reward / max(1, ep_decisions)
        return avg_reward, last_info

    # ------------------------------------------------------------------

    def _get_dag_tensors(self, agent_id: int):
        """Return DAG tensors cached for the given agent during episode encoding."""
        if self._dag_tensors and agent_id < len(self._dag_tensors):
            return self._dag_tensors[agent_id]
        # Fallback: rebuild from env (should not trigger, but prevents crash)
        dag_x, dag_ei, res_x, res_ei = _build_graph_inputs_v2(self.env, agent_id)
        return (
            dag_x.to(self.device), dag_ei.to(self.device),
            res_x.to(self.device), res_ei.to(self.device),
        )

    def _ppo_update(self) -> dict:
        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0

        for _ in range(self.cfg.ppo_epochs):
            batch = self.global_buffer.sample(self.cfg.mini_batch_size)

            obs_t        = torch.tensor(batch["obs"],        dtype=torch.float32, device=self.device)
            actions_t    = torch.tensor(batch["actions"],    dtype=torch.float32, device=self.device)
            h_pi_t       = torch.tensor(batch["h_pi"],       dtype=torch.float32, device=self.device)
            h_V_t        = torch.tensor(batch["h_V"],        dtype=torch.float32, device=self.device)
            global_obs_t = torch.tensor(batch["global_obs"], dtype=torch.float32, device=self.device)
            log_probs_old_t = torch.tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
            task_ids_t   = torch.tensor(batch["task_ids"],   dtype=torch.long,    device=self.device)
            agent_ids_t  = torch.tensor(batch["agent_ids"],  dtype=torch.long,    device=self.device)

            advantages_t = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device)
            returns_t    = torch.tensor(batch["returns"],    dtype=torch.float32, device=self.device)
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # Fix A: accumulate all agent losses, then single backward + step
            self.optimizer.zero_grad()

            epoch_actor  = 0.0
            epoch_critic = 0.0
            epoch_ent    = 0.0

            for agent_id, agent in enumerate(self.agents):
                # Fix B: use this agent's own DAG encoding
                dag_x, dag_ei, res_x, res_ei = self._get_dag_tensors(agent_id)

                self.shared_encoder.train()
                agent.actor.train()
                agent.critic.train()

                # Select only this agent's transitions for correct DAG matching
                mask = (agent_ids_t == agent_id)
                if mask.sum() == 0:
                    continue

                log_probs_new, entropies, values_new, _ = agent.evaluate_actions(
                    obs_t[mask], actions_t[mask], global_obs_t[mask],
                    h_pi_t[mask], h_V_t[mask],
                    dag_x, dag_ei, res_x, res_ei,
                    task_ids_t[mask],
                )

                ratio  = torch.exp(log_probs_new - log_probs_old_t[mask])
                surr1  = ratio * advantages_t[mask]
                surr2  = torch.clamp(ratio, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages_t[mask]
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values_new, returns_t[mask])
                entropy     = entropies.mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                loss.backward()   # accumulate gradients (no zero_grad here)

                epoch_actor  += actor_loss.item()
                epoch_critic += critic_loss.item()
                epoch_ent    += entropy.item()

            # Fix A: unified clip + step after all agents
            nn.utils.clip_grad_norm_(
                [p for pg in self.optimizer.param_groups for p in pg["params"]],
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()
            self._train_step += 1

            M = len(self.agents)
            total_actor_loss  += epoch_actor  / M
            total_critic_loss += epoch_critic / M
            total_entropy     += epoch_ent    / M

        n_epochs = self.cfg.ppo_epochs
        return {
            "actor_loss":  total_actor_loss  / n_epochs,
            "critic_loss": total_critic_loss / n_epochs,
            "entropy":     total_entropy     / n_epochs,
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
