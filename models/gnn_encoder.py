"""
Dual-path GNN encoder for AMAPPO.

Encodes two heterogeneous graphs:
  1. Task DAG: directed, GraphSAGE with separate upstream/downstream paths
  2. Resource graph: undirected R-node graph (local, UAVs, satellites, cloud)

Both produce 64-dim max-pooled graph embeddings that are concatenated and
projected to a 128-dim joint embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class TaskDAGEncoder(nn.Module):
    """GraphSAGE-based DAG encoder with separate upstream/downstream aggregation."""

    def __init__(self, in_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.conv_fwd1 = SAGEConv(in_dim, hidden_dim)
        self.conv_bwd1 = SAGEConv(in_dim, hidden_dim)
        self.conv_fwd2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv_bwd2 = SAGEConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index_rev = edge_index.flip(0)

        h_fwd = self.conv_fwd1(x, edge_index)
        h_bwd = self.conv_bwd1(x, edge_index_rev)
        h = F.relu(self.bn1(h_fwd + h_bwd))

        h_fwd = self.conv_fwd2(h, edge_index)
        h_bwd = self.conv_bwd2(h, edge_index_rev)
        h = F.relu(self.bn2(h_fwd + h_bwd))

        return h.max(dim=0).values


class ResourceGraphEncoder(nn.Module):
    """SAGEConv encoder for the fine-grained computing resource graph."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (R, 2) node feature matrix [current_load, capacity_norm]
            edge_index: (2, E) undirected edges (both directions expected)
        Returns:
            (64,) graph-level embedding via max-pooling over all nodes
        """
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        return h.max(dim=0).values


class GNNEncoder(nn.Module):
    """Dual-path GNN encoder combining task DAG and resource graph embeddings."""

    def __init__(self):
        super().__init__()
        self.dag_encoder = TaskDAGEncoder(in_dim=5, hidden_dim=64)
        self.res_encoder = ResourceGraphEncoder(in_dim=2, hidden_dim=64)
        self.joint_proj = nn.Linear(128, 128)
        self.out_dim = 128

    def forward(
        self,
        dag_x: torch.Tensor,           # (N, 5)
        dag_edge_index: torch.Tensor,  # (2, E)
        res_x: torch.Tensor,           # (R, 2)
        res_edge_index: torch.Tensor,  # (2, E_res)
    ) -> torch.Tensor:
        dag_emb = self.dag_encoder(dag_x, dag_edge_index)
        res_emb = self.res_encoder(res_x, res_edge_index)

        joint = torch.cat([dag_emb, res_emb], dim=0)
        return F.relu(self.joint_proj(joint))


if __name__ == "__main__":
    import networkx as nx

    G = nx.path_graph(5, create_using=nx.DiGraph)
    x = torch.randn(5, 5)
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    n_resources = 7
    res_x = torch.randn(n_resources, 2)
    res_edges = [[i, j] for i in range(n_resources) for j in range(n_resources) if i != j]
    res_edge_index = torch.tensor(res_edges, dtype=torch.long).t().contiguous()

    encoder = GNNEncoder()
    out = encoder(x, edge_index, res_x, res_edge_index)
    print("Output shape:", out.shape)
    assert out.shape == (128,), f"Expected (128,), got {out.shape}"

    out.sum().backward()
    print("Gradient flow: OK")
    print("PASS")
