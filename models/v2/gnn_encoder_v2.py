import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class _BidirectionalSAGELayer(nn.Module):
    """Single bidirectional SAGE layer: separate upstream/downstream, concat + BN + ReLU."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        half = out_dim // 2
        self.conv_fwd = SAGEConv(in_dim, half)
        self.conv_bwd = SAGEConv(in_dim, half)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index_rev = edge_index.flip(0)
        h_fwd = self.conv_fwd(x, edge_index)        # (N, half)
        h_bwd = self.conv_bwd(x, edge_index_rev)    # (N, half)
        h = torch.cat([h_fwd, h_bwd], dim=-1)       # (N, out_dim)
        # Skip BatchNorm when N=1 (single node, no neighbors)
        if h.shape[0] > 1:
            h = self.bn(h)
        return F.relu(h)


class TaskDAGEncoderV2(nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.layer1 = _BidirectionalSAGELayer(in_dim, hidden_dim)
        self.layer2 = _BidirectionalSAGELayer(hidden_dim, hidden_dim)
        self.graph_enc_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Returns:
            node_embs: (N, 64)
            graph_enc: (64,)
        """
        h = self.layer1(x, edge_index)                # (N, 64)
        node_embs = self.layer2(h, edge_index)        # (N, 64)

        projected = F.relu(self.graph_enc_proj(node_embs))  # (N, 64)
        graph_enc = projected.max(dim=0).values              # (64,)
        return node_embs, graph_enc


class ResourceGraphEncoderV2(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns: server_embs (4, 64)"""
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        return h  # (4, 64)


class GNNEncoderV2(nn.Module):
    def __init__(self, dag_in_dim: int = 5, res_in_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.dag_encoder = TaskDAGEncoderV2(dag_in_dim, hidden_dim)
        self.res_encoder = ResourceGraphEncoderV2(res_in_dim, hidden_dim)
        self.out_dim = hidden_dim

    def forward(
        self,
        dag_x: torch.Tensor,           # (N, 5)
        dag_edge_index: torch.Tensor,  # (2, E)
        res_x: torch.Tensor,           # (4, 2)
        res_edge_index: torch.Tensor,  # (2, E_res)
    ):
        """
        Returns:
            node_embs:   (N, 64)   task node embeddings
            server_embs: (4, 64)   server node embeddings
            graph_enc:   (64,)     global graph encoding
        """
        node_embs, graph_enc = self.dag_encoder(dag_x, dag_edge_index)
        server_embs = self.res_encoder(res_x, res_edge_index)
        return node_embs, server_embs, graph_enc
