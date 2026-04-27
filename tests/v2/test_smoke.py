import torch
import numpy as np
from utils.config import Config
from models.v2.gnn_encoder_v2 import GNNEncoderV2
from models.v2.actor_v2 import ActorV2
from models.v2.agent_v2 import MAPPOAgentV2


def test_full_encode_act_cycle():
    cfg = Config()
    cfg.device = "cpu"
    cfg.sync_derived_fields()
    encoder = GNNEncoderV2()

    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg, shared_encoder=encoder)

    n = 5
    dag_x = torch.randn(n, 5)
    edges = [[i, i + 1] for i in range(n - 1)]
    dag_ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    res_x = torch.randn(cfg.resource_node_count, 2)
    res_edges = [[i, j] for i in range(cfg.resource_node_count) for j in range(cfg.resource_node_count) if i != j]
    res_ei = torch.tensor(res_edges, dtype=torch.long).t().contiguous()

    agent.encode(dag_x, dag_ei, res_x, res_ei)

    for step in range(n):
        obs = np.random.randn(cfg.obs_dim).astype(np.float32)
        action, log_prob, h_pi = agent.act(obs)
        assert action.shape == (7,), f"step {step}: {action.shape}"

    agent.reset_episode()
    assert agent.step_idx == 0
