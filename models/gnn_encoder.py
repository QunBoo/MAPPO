"""
Dual-path GNN encoder for AMAPPO.

Encodes two heterogeneous graphs:
  1. Task DAG  — directed, GraphSAGE with separate upstream/downstream paths
  2. Resource Graph — undirected 4-node graph (local, UAV, satellite, cloud)

Both produce 64-dim max-pooled graph embeddings that are concatenated and
projected to a 128-dim joint embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class TaskDAGEncoder(nn.Module):
    """GraphSAGE-based DAG encoder with separate upstream/downstream aggregation.

    The DAG carries directional information: upstream (predecessor) edges
    propagate information *into* a node; downstream (successor) edges
    propagate information *out of* a node.  We handle this by running one
    SAGEConv on the original edge_index (forward / downstream direction) and a
    second SAGEConv on the reversed edge_index (backward / upstream direction),
    then summing the two streams at each layer.
    """

    def __init__(self, in_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        # Layer 1 — forward (downstream) and backward (upstream) convolutions
        self.conv_fwd1 = SAGEConv(in_dim, hidden_dim)
        self.conv_bwd1 = SAGEConv(in_dim, hidden_dim)
        # Layer 2 — same dual structure on the hidden representation
        self.conv_fwd2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv_bwd2 = SAGEConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (N, 5)  node feature matrix
            edge_index: (2, E)  directed edges  [src, dst]
        Returns:
            (64,)  graph-level embedding via max-pooling over all nodes
        """
        # Build reversed edge_index for upstream (predecessor) aggregation
        edge_index_rev = edge_index.flip(0)  # swap src <-> dst

        # --- Layer 1 ---
        h_fwd = self.conv_fwd1(x, edge_index)
        h_bwd = self.conv_bwd1(x, edge_index_rev)
        h = h_fwd + h_bwd                     # combine both directions
        h = self.bn1(h)
        h = F.relu(h)

        # --- Layer 2 ---
        h_fwd = self.conv_fwd2(h, edge_index)
        h_bwd = self.conv_bwd2(h, edge_index_rev)
        h = h_fwd + h_bwd
        h = self.bn2(h)
        h = F.relu(h)                          # (N, 64)

        # Max-pooling over all nodes → graph-level embedding
        graph_emb = h.max(dim=0).values        # (64,)
        return graph_emb


class ResourceGraphEncoder(nn.Module):
    """SAGEConv encoder for the computing resource graph.

    The resource graph has exactly 4 nodes (local device, UAV, satellite, cloud)
    connected by undirected edges.  A fully-connected topology is expected from
    the caller, but the encoder works with any supplied edge_index.
    """

    def __init__(self, in_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (4, 2)  node feature matrix [cpu_freq_norm, current_load]
            edge_index: (2, E)  undirected edges (both directions expected)
        Returns:
            (64,)  graph-level embedding via max-pooling over all nodes
        """
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)                  # (4, 64)

        graph_emb = h.max(dim=0).values  # (64,)
        return graph_emb


class GNNEncoder(nn.Module):
    """Dual-path GNN encoder combining task DAG and resource graph embeddings.

    Architecture:
        TaskDAGEncoder      → 64-dim
        ResourceGraphEncoder → 64-dim
        concat              → 128-dim
        Linear(128, 128) + ReLU → 128-dim  (joint embedding)
    """

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
        res_x: torch.Tensor,           # (4, 2)
        res_edge_index: torch.Tensor,  # (2, E_res)
    ) -> torch.Tensor:
        """
        Returns:
            (128,)  joint embedding
        """
        dag_emb = self.dag_encoder(dag_x, dag_edge_index)   # (64,)
        res_emb = self.res_encoder(res_x, res_edge_index)   # (64,)

        joint = torch.cat([dag_emb, res_emb], dim=0)        # (128,)
        joint = F.relu(self.joint_proj(joint))               # (128,)
        return joint


# ---------------------------------------------------------------------------
# Quick verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import networkx as nx

    # ---- Task DAG: 5-node path graph ----
    G = nx.path_graph(5, create_using=nx.DiGraph)
    x = torch.randn(5, 5)          # 5 nodes, 5 features
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ---- Resource graph: 4 nodes, fully connected (undirected) ----
    res_x = torch.randn(4, 2)
    res_edges = [[i, j] for i in range(4) for j in range(4) if i != j]
    res_edge_index = torch.tensor(res_edges, dtype=torch.long).t().contiguous()

    encoder = GNNEncoder()
    out = encoder(x, edge_index, res_x, res_edge_index)
    print("Output shape:", out.shape)   # Should be torch.Size([128])
    assert out.shape == (128,), f"Expected (128,), got {out.shape}"

    # Test backward pass
    out.sum().backward()
    print("Gradient flow: OK")
    print("PASS")
