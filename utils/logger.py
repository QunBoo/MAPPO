import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False


class Logger:
    def __init__(self, log_dir: str, algo_name: str = "amappo"):
        self._log_dir = f"{log_dir}/{algo_name}"
        self.episode = 0

        if _TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self._log_dir)
        else:
            self.writer = None
            print(
                f"[Logger] TensorBoard is not available. "
                f"Metrics will be printed to console. "
                f"Install with: pip install tensorboard"
            )

    def log_episode(self, episode: int, metrics: dict):
        """
        Log episode-level metrics.

        metrics dict keys:
          - episode_reward (float)
          - T_total (float)
          - E_total (float)
          - cost (float)
          - violations (list of 5 ints)
        """
        self.episode = episode

        episode_reward = metrics.get("episode_reward", 0.0)
        T_total = metrics.get("T_total", 0.0)
        E_total = metrics.get("E_total", 0.0)
        cost = metrics.get("cost", 0.0)
        violations = metrics.get("violations", [0, 0, 0, 0, 0])
        total_violations = int(np.sum(violations))

        if self.writer is not None:
            self.writer.add_scalar("episode/reward", episode_reward, episode)
            self.writer.add_scalar("episode/T_total", T_total, episode)
            self.writer.add_scalar("episode/E_total", E_total, episode)
            self.writer.add_scalar("episode/cost", cost, episode)
            self.writer.add_scalar("episode/violations_total", total_violations, episode)
            for i, v in enumerate(violations):
                self.writer.add_scalar(f"episode/violation_{i}", v, episode)
        else:
            print(
                f"[Episode {episode}] reward={episode_reward:.4f}  "
                f"T_total={T_total:.4f}  E_total={E_total:.4f}  "
                f"cost={cost:.4f}  violations={violations}"
            )

    def log_training(self, step: int, metrics: dict):
        """
        Log training step metrics.

        metrics dict keys:
          - critic_loss (float)
          - actor_loss (float)
          - entropy (float)
        """
        critic_loss = metrics.get("critic_loss", 0.0)
        actor_loss = metrics.get("actor_loss", 0.0)
        entropy = metrics.get("entropy", 0.0)

        if self.writer is not None:
            self.writer.add_scalar("train/critic_loss", critic_loss, step)
            self.writer.add_scalar("train/actor_loss", actor_loss, step)
            self.writer.add_scalar("train/entropy", entropy, step)
        else:
            print(
                f"[Train step {step}] critic_loss={critic_loss:.6f}  "
                f"actor_loss={actor_loss:.6f}  entropy={entropy:.6f}"
            )

    def close(self):
        if self.writer is not None:
            self.writer.close()
