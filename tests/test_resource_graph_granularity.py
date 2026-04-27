import math
import subprocess
import sys
import numpy as np

from utils.config import Config
from env.sec_env import SECEnv
from algorithms.amappo_v2 import AMAPPOv2Trainer
from algorithms.mappo import MAPPOTrainer


def _make_cfg(**overrides):
    cfg = Config(
        N=12,
        M=2,
        K=3,
        J=5,
        max_steps=12,
        device="cpu",
        epochs=2,
        mini_batch_size=4,
        log_interval=1,
        save_interval=999999,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.sync_derived_fields()
    return cfg


def _action_for_offload(offload_idx: int, comp_logit: float = 0.0) -> np.ndarray:
    action = np.zeros(8, dtype=np.float32)
    action[offload_idx] = 10.0
    action[5] = comp_logit
    return action


def test_resource_graph_shapes_and_node_order():
    cfg = _make_cfg(M=2, K=3)
    env = SECEnv(cfg)
    env.reset()

    res_x, res_edge_index, node_meta = env.get_resource_graph_data(include_meta=True)
    expected_names = [
        "local",
        "uav_0",
        "uav_1",
        "sat_0",
        "sat_1",
        "sat_2",
        "cloud",
    ]

    assert cfg.resource_node_count == 7
    assert res_x.shape == (cfg.resource_node_count, 2)
    assert res_edge_index.shape == (2, cfg.resource_node_count * (cfg.resource_node_count - 1))
    assert [item["name"] for item in node_meta] == expected_names


def test_conda_appo_runtime_dependency_probe():
    result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "appo",
            "python",
            "-c",
            "import torch, torch_geometric; print('runtime-imports-ok')",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "runtime-imports-ok" in result.stdout


def test_observation_uses_fine_grained_server_states():
    cfg = _make_cfg(M=2, K=3)
    env = SECEnv(cfg)
    obs_dict = env.reset()

    obs = obs_dict[0]
    assert obs.shape == (cfg.obs_dim,)

    server_states = obs[25:-8]
    prev_action = obs[-8:]
    assert server_states.shape == (cfg.resource_node_count,)
    assert np.allclose(prev_action, 0.0)


def test_step_updates_current_uav_node_only():
    cfg = _make_cfg(M=2, K=2)
    env = SECEnv(cfg)
    env.reset()

    action_dict = {
        0: _action_for_offload(1),
        1: _action_for_offload(0),
    }
    env.step(action_dict)

    assert env.resource_loads[env.uav_resource_index(0)] > 0.0
    assert math.isclose(env.resource_loads[env.uav_resource_index(1)], 0.0, abs_tol=1e-8)
    assert np.allclose(env.resource_loads[env.first_sat_resource_index():env.cloud_resource_index()], 0.0)


def test_step_updates_nearest_satellite_node_only():
    cfg = _make_cfg(M=1, K=3)
    env = SECEnv(cfg)
    env.reset()

    nearest_sat = env._nearest_sat(0)
    env.step({0: _action_for_offload(2)})

    sat_slice = env.resource_loads[env.first_sat_resource_index():env.cloud_resource_index()]
    active_idx = nearest_sat
    assert sat_slice[active_idx] > 0.0
    for idx, load in enumerate(sat_slice):
        if idx != active_idx:
            assert math.isclose(load, 0.0, abs_tol=1e-8)


def test_step_updates_cloud_node_only():
    cfg = _make_cfg(M=1, K=2)
    env = SECEnv(cfg)
    env.reset()

    env.step({0: _action_for_offload(3)})

    assert env.resource_loads[env.cloud_resource_index()] > 0.0
    assert np.allclose(
        env.resource_loads[env.first_sat_resource_index():env.cloud_resource_index()],
        0.0,
    )
    assert math.isclose(env.resource_loads[env.local_resource_index()], 0.0, abs_tol=1e-8)


def test_amappov2_run_episode_smoke_with_fine_grained_resources():
    cfg = _make_cfg(M=2, K=3, mini_batch_size=4)
    trainer = AMAPPOv2Trainer(cfg)

    avg_reward, info = trainer._run_episode()

    assert np.isfinite(avg_reward)
    assert np.isfinite(info["T_total"])
    assert np.isfinite(info["E_total"])
    assert len(trainer.global_buffer) > 0


def test_mappo_run_episode_smoke_with_fine_grained_resources():
    cfg = _make_cfg(M=2, K=3, mini_batch_size=4)
    trainer = MAPPOTrainer(cfg)

    avg_reward, info = trainer._run_episode()

    assert np.isfinite(avg_reward)
    assert np.isfinite(info["T_total"])
    assert np.isfinite(info["E_total"])
    assert len(trainer.global_buffer) > 0


def test_amappov2_cli_smoke_with_fine_grained_resources():
    result = subprocess.run(
        [
            sys.executable,
            "experiments/train_v2.py",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--log_interval",
            "1",
            "--save_interval",
            "999999",
            "--mini_batch_size",
            "32",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert "[AMAPPOv2] ep=" in result.stdout
    assert "Training complete" in result.stdout


def test_mappo_cli_smoke_with_fine_grained_resources():
    result = subprocess.run(
        [
            sys.executable,
            "experiments/train.py",
            "--algo",
            "mappo",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--log_interval",
            "1",
            "--save_interval",
            "999999",
            "--mini_batch_size",
            "32",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert "[MAPPO] ep=" in result.stdout
