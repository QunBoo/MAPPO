import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Transition:
    """Single transition for one agent at one timestep."""
    obs: np.ndarray        # (37,) observation
    action: np.ndarray     # (8,) action
    reward: float          # scalar reward
    h_pi: np.ndarray       # (1, 1, 64) actor GRU hidden state
    h_V: np.ndarray        # (1, 1, 64) critic GRU hidden state
    global_obs: np.ndarray # (148,) global state (all agents concatenated)
    done: bool             # episode done flag


class AgentBuffer:
    """Per-agent transition cache ξ_i."""

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.transitions: List[Transition] = []

    def add(self, transition: Transition):
        self.transitions.append(transition)

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)


class GlobalBuffer:
    """Global experience pool MB. Collects from all agent buffers."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._obs: list = []
        self._actions: list = []
        self._rewards: list = []
        self._h_pi: list = []
        self._h_V: list = []
        self._global_obs: list = []
        self._dones: list = []

    def add_from_agent_buffer(self, agent_buffer: AgentBuffer):
        """Pour all transitions from an agent buffer into the global pool."""
        for t in agent_buffer.transitions:
            self._obs.append(t.obs)
            self._actions.append(t.action)
            self._rewards.append(t.reward)
            self._h_pi.append(t.h_pi)
            self._h_V.append(t.h_V)
            self._global_obs.append(t.global_obs)
            self._dones.append(float(t.done))

        # Trim to capacity (oldest first)
        if len(self._obs) > self.capacity:
            excess = len(self._obs) - self.capacity
            self._obs = self._obs[excess:]
            self._actions = self._actions[excess:]
            self._rewards = self._rewards[excess:]
            self._h_pi = self._h_pi[excess:]
            self._h_V = self._h_V[excess:]
            self._global_obs = self._global_obs[excess:]
            self._dones = self._dones[excess:]

    def sample(self, batch_size: int = 128) -> Dict[str, np.ndarray]:
        """Sample a mini-batch. Returns dict with numpy arrays."""
        N = len(self._obs)
        if N == 0:
            raise ValueError("Buffer is empty")
        indices = np.random.choice(N, size=min(batch_size, N), replace=False)
        return {
            'obs':        np.array([self._obs[i]        for i in indices]),  # (B, 37)
            'actions':    np.array([self._actions[i]    for i in indices]),  # (B, 8)
            'rewards':    np.array([self._rewards[i]    for i in indices]),  # (B,)
            'h_pi':       np.array([self._h_pi[i]       for i in indices]),  # (B, 1, 1, 64)
            'h_V':        np.array([self._h_V[i]        for i in indices]),  # (B, 1, 1, 64)
            'global_obs': np.array([self._global_obs[i] for i in indices]),  # (B, 148)
            'dones':      np.array([self._dones[i]      for i in indices]),  # (B,)
        }

    def clear(self):
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._h_pi.clear()
        self._h_V.clear()
        self._global_obs.clear()
        self._dones.clear()

    def __len__(self):
        return len(self._obs)

    def compute_returns_and_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple:
        """
        Compute GAE advantages and returns.

        All inputs are 1-D numpy arrays of length T.
        Returns: (advantages, returns) both shape (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            next_value = 0.0 if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns
