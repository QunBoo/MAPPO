"""
Asynchronous AMAPPO trainer.

Extends synchronous MAPPO with a dual-clock mechanism:
  - Global clock t' = 1..GT: drives the main loop.
  - Per-agent local clock t_i: the earliest global step at which agent i is
    "available" (i.e., the last task has finished).

At each global timestep:
  - Available agents (t' >= t_i) act, record experience, and schedule their
    next availability based on the sampled task execution time.
  - Unavailable agents skip — no experience is written.

This means different agents act at different frequencies, reflecting real
asynchronous execution in the physical system.
"""

from __future__ import annotations

import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from utils.config import Config
from utils.buffer import AgentBuffer, GlobalBuffer, Transition
from utils.logger import Logger
from models.agent import MAPPOAgent
from models.gnn_encoder import GNNEncoder
from env.sec_env import SECEnv
from algorithms.mappo import _build_graph_inputs   # reuse graph-input helper


class AMAPPOTrainer:
    """Asynchronous MAPPO (AMAPPO) trainer with dual-clock mechanism."""

    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)

        # Shared GNN encoder (shared across all agents)
        self.shared_gnn = GNNEncoder().to(self.device)

        # Agents
        self.agents: List[MAPPOAgent] = [
            MAPPOAgent(agent_id=m, config=config, shared_gnn_encoder=self.shared_gnn)
            for m in range(config.M)
        ]

        # Unified optimizer
        all_params = list(self.shared_gnn.parameters())
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
        self.logger = Logger(log_dir=config.log_dir, algo_name="amappo")
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self._train_step = 0

        # Decision frequency tracking (for diagnostics)
        self._agent_decision_count: np.ndarray = np.zeros(config.M, dtype=int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self):
        """Run full training for config.epochs episodes."""
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
                    f"[AMAPPO] ep={ep:4d}  reward={ep_reward:8.3f}  "
                    f"T={ep_info.get('T_total',0):.4f}  "
                    f"E={ep_info.get('E_total',0):.4e}  "
                    f"decisions={decisions_per_agent}"
                )
                self._agent_decision_count[:] = 0

            if ep % self.cfg.save_interval == 0:
                self._save_checkpoint(ep)

        self.logger.close()
        print("[AMAPPO] Training complete.")

    # ------------------------------------------------------------------
    # Episode rollout
    # ------------------------------------------------------------------

    def _run_episode(self) -> Tuple[float, dict]:
        """
        Collect one episode of experience with the dual-clock async mechanism.
        """
        obs_dict = self.env.reset()

        for agent in self.agents:
            agent.reset_hidden()
        for buf in self.agent_buffers:
            buf.clear()

        # Per-agent local clocks: all start available at step 0
        # t_i = global step at which agent i becomes available again
        local_clocks: np.ndarray = np.zeros(self.cfg.M, dtype=int)

        total_reward = 0.0
        last_info: dict = {}
        global_step = 0

        # We need to track last obs per agent for the unavailable-agent steps
        last_obs: Dict[int, np.ndarray] = {m: obs_dict[m] for m in range(self.cfg.M)}

        # Build global obs helper
        def _global_obs() -> np.ndarray:
            return np.concatenate([last_obs[m] for m in range(self.cfg.M)], axis=0)

        # action_dict starts as zeros (unavailable agents re-use last action)
        action_dict: Dict[int, np.ndarray] = {m: np.zeros(self.cfg.action_dim) for m in range(self.cfg.M)}

        while True:
            global_obs = _global_obs()

            available_agents = [m for m in range(self.cfg.M) if global_step >= local_clocks[m]]

            for m in available_agents:
                agent = self.agents[m]
                obs_m = last_obs[m]

                dag_x, dag_ei, res_x, res_ei = _build_graph_inputs(self.env, m)
                action, log_prob, h_pi = agent.act(obs_m, dag_x, dag_ei, res_x, res_ei)
                value, h_V = agent.get_value(global_obs)

                action_dict[m] = action

                # Estimate task execution time to schedule next availability
                # Use a simple proxy: sample from a geometric distribution
                # with mean proportional to the task complexity
                task_node = self.env._current_task(m)
                if task_node is not None:
                    c_cycles = self.env.dags[m].nodes[task_node].get("C", 1.0)
                    # Mean execution slots ≈ C / (some base rate)
                    mean_slots = max(1, int(c_cycles / 0.5))
                    exec_slots = max(1, np.random.geometric(1.0 / mean_slots))
                else:
                    exec_slots = 1

                local_clocks[m] = global_step + exec_slots
                self._agent_decision_count[m] += 1

                # Store in agent buffer
                done_flag = False   # will be updated after env.step
                t = Transition(
                    obs=obs_m,
                    action=action,
                    reward=0.0,    # placeholder; updated after env.step
                    h_pi=h_pi.detach().cpu().numpy(),
                    h_V=h_V.detach().cpu().numpy(),
                    global_obs=global_obs.copy(),
                    done=done_flag,
                )
                self.agent_buffers[m].add(t)

            # Step the environment with current action_dict
            next_obs_dict, rew_dict, done, info = self.env.step(action_dict)
            last_info = info

            # Update rewards in the just-added transitions (only available agents)
            for m in available_agents:
                reward = rew_dict.get(m, 0.0)
                total_reward += reward
                # Update the last transition's reward and done flag
                buf = self.agent_buffers[m]
                if len(buf.transitions) > 0:
                    buf.transitions[-1].reward = reward
                    buf.transitions[-1].done   = done

            # Update last observations
            for m in range(self.cfg.M):
                last_obs[m] = next_obs_dict[m]

            global_step += 1

            if done:
                break

        # Pour agent buffers into global buffer
        for buf in self.agent_buffers:
            self.global_buffer.add_from_agent_buffer(buf)
            buf.clear()

        avg_reward = total_reward / max(1, sum(self._agent_decision_count))
        return avg_reward, last_info

    # ------------------------------------------------------------------
    # PPO update (same logic as MAPPO)
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

        B = obs_t.shape[0]

        dag_x, dag_ei, res_x, res_ei = _build_graph_inputs(self.env, 0)
        dag_x  = dag_x.to(self.device)
        dag_ei = dag_ei.to(self.device)
        res_x  = res_x.to(self.device)
        res_ei = res_ei.to(self.device)

        # Compute baseline values for GAE
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

        agent = self.agents[0]
        self.shared_gnn.train()
        agent.actor.train()
        agent.critic.train()

        log_probs_new, entropies, values_new, _ = agent.evaluate_actions(
            obs_t, actions_t, global_obs_t, h_pi_t, h_V_t,
            dag_x, dag_ei, res_x, res_ei,
        )

        log_probs_old = log_probs_new.detach()
        ratio  = torch.exp(log_probs_new - log_probs_old)
        surr1  = ratio * advantages_t
        surr2  = torch.clamp(ratio, 1.0 - self.cfg.eps_clip, 1.0 + self.cfg.eps_clip) * advantages_t
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
        path = os.path.join(self.cfg.checkpoint_dir, f"amappo_ep{episode}.pt")
        state = {
            "episode": episode,
            "shared_gnn": self.shared_gnn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        for m, agent in enumerate(self.agents):
            state[f"actor_{m}"]  = agent.actor.state_dict()
            state[f"critic_{m}"] = agent.critic.state_dict()
        torch.save(state, path)
        print(f"[AMAPPO] Checkpoint saved: {path}")
