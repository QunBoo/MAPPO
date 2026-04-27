import pytest
import torch
from models.v2.gnn_encoder_v2 import GNNEncoderV2


def _make_dag(n_nodes=5):
    """Chain DAG: 0->1->2->3->4"""
    x = torch.randn(n_nodes, 5)
    edges = [[i, i + 1] for i in range(n_nodes - 1)]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Single node: empty 2D edge_index
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return x, edge_index


def _make_res(n_resources=7):
    x = torch.randn(n_resources, 2)
    edges = [[i, j] for i in range(n_resources) for j in range(n_resources) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


def test_output_shapes():
    encoder = GNNEncoderV2()
    dag_x, dag_ei = _make_dag(5)
    res_x, res_ei = _make_res()

    node_embs, server_embs, graph_enc = encoder(dag_x, dag_ei, res_x, res_ei)

    assert node_embs.shape == (5, 64), f"node_embs: {node_embs.shape}"
    assert server_embs.shape == (7, 64), f"server_embs: {server_embs.shape}"
    assert graph_enc.shape == (64,), f"graph_enc: {graph_enc.shape}"


def test_different_dag_sizes():
    encoder = GNNEncoderV2()
    res_x, res_ei = _make_res()

    for n in [1, 3, 10, 20]:
        dag_x, dag_ei = _make_dag(n)
        node_embs, server_embs, graph_enc = encoder(dag_x, dag_ei, res_x, res_ei)
        assert node_embs.shape == (n, 64), f"n={n}: {node_embs.shape}"


def test_gradient_flow():
    encoder = GNNEncoderV2()
    dag_x, dag_ei = _make_dag(5)
    res_x, res_ei = _make_res()

    node_embs, server_embs, graph_enc = encoder(dag_x, dag_ei, res_x, res_ei)
    loss = node_embs.sum() + server_embs.sum() + graph_enc.sum()
    loss.backward()

    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_node_embs_are_distinct():
    """Different nodes should produce different embeddings (non-degenerate)."""
    encoder = GNNEncoderV2()
    dag_x = torch.eye(5, 5)   # distinct inputs
    edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    res_x, res_ei = _make_res()

    node_embs, _, _ = encoder(dag_x, edge_index, res_x, res_ei)
    for i in range(5):
        for j in range(i + 1, 5):
            assert not torch.allclose(node_embs[i], node_embs[j]), \
                f"nodes {i} and {j} have identical embeddings"
