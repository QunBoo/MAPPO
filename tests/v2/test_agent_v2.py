import pytest
import torch
import numpy as np
from models.v2.agent_v2 import MAPPOAgentV2
from utils.config import Config


def _make_graph(cfg: Config, n=5):
    dag_x = torch.randn(n, 5)
    # Chain DAG: 0->1->2->3->4
    edges = [[i, i + 1] for i in range(n - 1)]
    dag_ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    res_x = torch.randn(cfg.resource_node_count, 2)
    res_edges = [[i, j] for i in range(cfg.resource_node_count) for j in range(cfg.resource_node_count) if i != j]
    res_ei = torch.tensor(res_edges, dtype=torch.long).t().contiguous()
    return dag_x, dag_ei, res_x, res_ei


def _make_obs(cfg: Config):
    return np.random.randn(cfg.obs_dim).astype(np.float32)


def test_encode_fills_cache():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(cfg, 5)

    assert agent.node_embs is None
    agent.encode(dag_x, dag_ei, res_x, res_ei)
    assert agent.node_embs is not None
    assert agent.node_embs.shape == (5, 64)
    assert agent.server_embs.shape == (cfg.resource_node_count, 64)
    assert agent.graph_enc.shape == (64,)
    assert agent.topo_order is not None
    assert len(agent.topo_order) == 5
    assert agent.h_pi is not None   # initialized from graph_enc, not None


def test_topo_order_is_valid():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(cfg, 5)
    agent.encode(dag_x, dag_ei, res_x, res_ei)

    topo = agent.topo_order
    # Chain DAG: topo order must be [0,1,2,3,4]
    assert topo == list(range(5)), f"Expected [0,1,2,3,4], got {topo}"


def test_act_returns_correct_leo_shape():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(cfg, 5)
    agent.encode(dag_x, dag_ei, res_x, res_ei)

    obs = _make_obs(cfg)
    action, log_prob, h_pi = agent.act(obs)

    assert action.shape == (7,), f"LEO action: {action.shape}"
    assert isinstance(log_prob, float)
    assert h_pi.shape == (1, 1, 64)


def test_act_increments_step_idx():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(cfg, 5)
    agent.encode(dag_x, dag_ei, res_x, res_ei)

    assert agent.step_idx == 0
    agent.act(_make_obs(cfg))
    assert agent.step_idx == 1
    agent.act(_make_obs(cfg))
    assert agent.step_idx == 2


def test_reset_episode_clears_cache():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(cfg, 5)
    agent.encode(dag_x, dag_ei, res_x, res_ei)
    agent.act(_make_obs(cfg))

    agent.reset_episode()

    assert agent.node_embs is None
    assert agent.topo_order is None
    assert agent.step_idx == 0
    assert len(agent.decisions) == 0
    assert agent.h_pi is None


def test_uav_act_returns_correct_shape():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="UAV", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(cfg, 3)
    agent.encode(dag_x, dag_ei, res_x, res_ei)

    action, log_prob, h_pi = agent.act(_make_obs(cfg))
    assert action.shape == (9,), f"UAV action: {action.shape}"
