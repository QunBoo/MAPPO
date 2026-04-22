# AMAPPOv2 Implementation Plan
根据docs\superpowers\plans\2026-04-22-amappov2-implementation-plan.md的AMAPPOv2实现计划，开展AMAPPOv2算法的开发。

计划概览：7 个任务，按依赖顺序排列

任务	产物	测试数量
Task 1	子包骨架 models/v2/ + 测试目录	—
Task 2	GNNEncoderV2（双向拼接 + 节点级输出）	4 个
Task 3	ActorV2（GRU + 注意力 + LEO/UAV 动作头）	7 个
Task 4	MAPPOAgentV2（encode/act/reset）	6 个
Task 5	AMAPPOv2Trainer	导入验证
Task 6	experiments/train_v2.py	导入验证
Task 7	整合验证 + smoke test	1 个

使用Subagent-Driven方式执行 — 每个任务派发独立子 agent 执行，任务间有 review 检查点，迭代快

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改动任何现有文件的前提下，新建 `models/v2/` 子包和 `algorithms/amappo_v2.py`，实现严格对齐论文的编码器-解码器架构（节点级嵌入 + GRU + 注意力机制 + 按 agent_type 分类动作头）。

**Architecture:** `GNNEncoderV2` 返回节点级嵌入（而非图级嵌入），`ActorV2` 在 GRU 输出后加注意力机制生成上下文向量，`MAPPOAgentV2` 在 episode 开始时调用 `encode()` 缓存嵌入，每步 `act()` 复用缓存，避免重复运行编码器。`AMAPPOv2Trainer` 在原 `AMAPPOTrainer` 骨架上插入一行 `agent.encode()` 调用。

**Tech Stack:** Python 3.10+, PyTorch, torch_geometric (SAGEConv), networkx（拓扑排序）

---

## 文件映射

| 文件 | 状态 | 职责 |
|------|------|------|
| `models/v2/__init__.py` | 新建 | 子包入口 |
| `models/v2/gnn_encoder_v2.py` | 新建 | 双向拼接编码器，返回节点嵌入 + GraphEnc |
| `models/v2/actor_v2.py` | 新建 | GRU + 注意力 + 按 agent_type 分类动作头 |
| `models/v2/agent_v2.py` | 新建 | encode/act/evaluate 接口，episode 级缓存管理 |
| `algorithms/amappo_v2.py` | 新建 | AMAPPOv2Trainer，插入 encode() 调用 |
| `experiments/train_v2.py` | 新建 | 独立训练入口 |
| `tests/v2/test_gnn_encoder_v2.py` | 新建 | 编码器单元测试 |
| `tests/v2/test_actor_v2.py` | 新建 | ActorV2 单元测试 |
| `tests/v2/test_agent_v2.py` | 新建 | MAPPOAgentV2 单元测试 |
| `utils/config.py` | **不改动** | 只新增 `ConfigV2` dataclass 在 `amappo_v2.py` 内部定义 |

> **隔离原则**：`models/gnn_encoder.py`、`models/actor.py`、`models/agent.py`、`algorithms/amappo.py`、`experiments/train.py` 均**不修改**。

---

## Task 1：创建子包骨架与测试目录

**Files:**
- Create: `models/v2/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/v2/__init__.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p models/v2
mkdir -p tests/v2
```

- [ ] **Step 2: 写 `models/v2/__init__.py`**

```python
# models/v2/__init__.py
from models.v2.gnn_encoder_v2 import GNNEncoderV2
from models.v2.actor_v2 import ActorV2
from models.v2.agent_v2 import MAPPOAgentV2

__all__ = ["GNNEncoderV2", "ActorV2", "MAPPOAgentV2"]
```

- [ ] **Step 3: 写空 `__init__.py` 文件**

```bash
touch tests/__init__.py
touch tests/v2/__init__.py
```

- [ ] **Step 4: 验证目录结构**

运行：`python -c "import models.v2"`
预期：无报错（因为 `gnn_encoder_v2` 等还没有，此步会失败，可跳过，Task 2 完成后再验证）

- [ ] **Step 5: Commit**

```bash
git add models/v2/__init__.py tests/__init__.py tests/v2/__init__.py
git commit -m "feat(v2): scaffold models/v2 subpackage and tests directory"
```

---

## Task 2：实现 GNNEncoderV2

**Files:**
- Create: `models/v2/gnn_encoder_v2.py`
- Create: `tests/v2/test_gnn_encoder_v2.py`

### 核心设计要点
- 每层每路输出 `hidden_dim // 2 = 32` 维，拼接后恢复 `hidden_dim = 64` 维
- 保留 `BatchNorm`（与现有编码器一致）
- 全局图编码：`max_pool(FC(node_embs))`（先 FC 投影再 max-pool）
- 返回三元组：`(node_embs, server_embs, graph_enc)`

- [ ] **Step 1: 写失败测试**

`tests/v2/test_gnn_encoder_v2.py`:

```python
import pytest
import torch
from models.v2.gnn_encoder_v2 import GNNEncoderV2


def _make_dag(n_nodes=5):
    """Chain DAG: 0→1→2→3→4"""
    x = torch.randn(n_nodes, 5)
    edges = [[i, i + 1] for i in range(n_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


def _make_res():
    x = torch.randn(4, 2)
    edges = [[i, j] for i in range(4) for j in range(4) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


def test_output_shapes():
    encoder = GNNEncoderV2()
    dag_x, dag_ei = _make_dag(5)
    res_x, res_ei = _make_res()

    node_embs, server_embs, graph_enc = encoder(dag_x, dag_ei, res_x, res_ei)

    assert node_embs.shape == (5, 64), f"node_embs: {node_embs.shape}"
    assert server_embs.shape == (4, 64), f"server_embs: {server_embs.shape}"
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
    """不同节点应产生不同嵌入（非退化）"""
    encoder = GNNEncoderV2()
    dag_x = torch.eye(5, 5)   # 不同输入
    edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    res_x, res_ei = _make_res()

    node_embs, _, _ = encoder(dag_x, edge_index, res_x, res_ei)
    # 任意两行不完全相同
    for i in range(5):
        for j in range(i + 1, 5):
            assert not torch.allclose(node_embs[i], node_embs[j]), \
                f"nodes {i} and {j} have identical embeddings"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/v2/test_gnn_encoder_v2.py -v
```

预期：`ImportError: cannot import name 'GNNEncoderV2'`

- [ ] **Step 3: 实现 `GNNEncoderV2`**

`models/v2/gnn_encoder_v2.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class _BidirectionalSAGELayer(nn.Module):
    """单层双向 SAGE：分别聚合上游/下游，拼接后 BN+ReLU。"""

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
        return F.relu(self.bn(h))


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
        h = self.layer1(x, edge_index)       # (N, 64)
        node_embs = self.layer2(h, edge_index)  # (N, 64)

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
            node_embs:   (N, 64)   任务节点嵌入
            server_embs: (4, 64)   服务器节点嵌入
            graph_enc:   (64,)     全局图编码
        """
        node_embs, graph_enc = self.dag_encoder(dag_x, dag_edge_index)
        server_embs = self.res_encoder(res_x, res_edge_index)
        return node_embs, server_embs, graph_enc
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/v2/test_gnn_encoder_v2.py -v
```

预期：4 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add models/v2/gnn_encoder_v2.py tests/v2/test_gnn_encoder_v2.py
git commit -m "feat(v2): implement GNNEncoderV2 with bidirectional concat and node-level output"
```

---

## Task 3：实现 ActorV2

**Files:**
- Create: `models/v2/actor_v2.py`
- Create: `tests/v2/test_actor_v2.py`

### 核心设计要点
- GRU 输入：`concat([h_v_t(64), L_us_agg(64), server_agg(64), a_prev(8)])` = 200 维
- GRU 隐藏维度：64
- 注意力：`w_t @ node_embs.T → softmax → 加权求和 → concat([w_t, context])` = 128 维上下文
- LEO 动作头：`discrete(4) + continuous(3)` = 7 维输出
- UAV 动作头：`discrete(4) + continuous(5)` = 9 维输出
- `a_prev` 输入维度统一为 8（与现有 buffer 格式兼容）

- [ ] **Step 1: 写失败测试**

`tests/v2/test_actor_v2.py`:

```python
import pytest
import torch
from models.v2.actor_v2 import ActorV2


def _make_inputs(n_tasks=5, batch=None):
    if batch is None:
        h_v_t = torch.randn(64)
        L_us_agg = torch.randn(64)
        server_agg = torch.randn(64)
        a_prev = torch.randn(8)
        h_prev = None
        node_embs = torch.randn(n_tasks, 64)
    else:
        h_v_t = torch.randn(batch, 64)
        L_us_agg = torch.randn(batch, 64)
        server_agg = torch.randn(batch, 64)
        a_prev = torch.randn(batch, 8)
        h_prev = torch.zeros(1, batch, 64)
        node_embs = torch.randn(n_tasks, 64)
    return h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs


def test_leo_forward_shapes():
    actor = ActorV2(agent_type="LEO")
    h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs = _make_inputs()

    action, log_prob, h_next = actor(h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs)

    assert action.shape == (7,), f"LEO action: {action.shape}"
    assert log_prob.shape == (), f"log_prob: {log_prob.shape}"
    assert h_next.shape == (1, 1, 64), f"h_next: {h_next.shape}"


def test_uav_forward_shapes():
    actor = ActorV2(agent_type="UAV")
    h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs = _make_inputs()

    action, log_prob, h_next = actor(h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs)

    assert action.shape == (9,), f"UAV action: {action.shape}"
    assert log_prob.shape == (), f"log_prob: {log_prob.shape}"


def test_gru_hidden_state_propagates():
    actor = ActorV2(agent_type="LEO")
    h_v_t, L_us_agg, server_agg, a_prev, _, node_embs = _make_inputs()

    _, _, h1 = actor(h_v_t, L_us_agg, server_agg, a_prev, None, node_embs)
    _, _, h2 = actor(h_v_t, L_us_agg, server_agg, a_prev, h1, node_embs)

    assert not torch.allclose(h1, h2), "Hidden state should change across steps"


def test_evaluate_leo():
    actor = ActorV2(agent_type="LEO")
    B = 4
    h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs = _make_inputs(batch=B)
    discrete_actions = torch.randint(0, 4, (B,))

    log_probs, entropies, h_next = actor.evaluate(
        h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs, discrete_actions
    )

    assert log_probs.shape == (B,)
    assert entropies.shape == (B,)
    assert h_next.shape == (1, B, 64)


def test_evaluate_uav():
    actor = ActorV2(agent_type="UAV")
    B = 3
    h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs = _make_inputs(batch=B)
    discrete_actions = torch.randint(0, 4, (B,))

    log_probs, entropies, h_next = actor.evaluate(
        h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs, discrete_actions
    )

    assert log_probs.shape == (B,)
    assert entropies.shape == (B,)


def test_gradient_flow():
    actor = ActorV2(agent_type="LEO")
    h_v_t, L_us_agg, server_agg, a_prev, _, node_embs = _make_inputs()

    action, log_prob, h_next = actor(h_v_t, L_us_agg, server_agg, a_prev, None, node_embs)
    log_prob.backward()

    for name, p in actor.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_invalid_agent_type():
    with pytest.raises(ValueError, match="agent_type"):
        ActorV2(agent_type="INVALID")
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/v2/test_actor_v2.py -v
```

预期：`ImportError: cannot import name 'ActorV2'`

- [ ] **Step 3: 实现 `ActorV2`**

`models/v2/actor_v2.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Optional, Tuple

# GRU 输入维度：h_v_t(64) + L_us_agg(64) + server_agg(64) + a_prev(8) = 200
_GRU_INPUT_DIM = 200
_GRU_HIDDEN = 64
_NODE_EMB_DIM = 64
_CONTEXT_DIM = _GRU_HIDDEN + _NODE_EMB_DIM  # 128

# LEO: discrete(4) + continuous(3: z, P, f)
_LEO_DISC = 4
_LEO_CONT = 3
# UAV: discrete(4) + continuous(5: z, P, f_m, f_k, q)
_UAV_DISC = 4
_UAV_CONT = 5


class ActorV2(nn.Module):
    def __init__(self, agent_type: str = "LEO"):
        super().__init__()
        if agent_type not in ("LEO", "UAV"):
            raise ValueError(f"agent_type must be 'LEO' or 'UAV', got '{agent_type}'")

        self.agent_type = agent_type

        # GRU 初始化投影（用 graph_enc 初始化隐藏状态）
        self.hidden_init = nn.Linear(_NODE_EMB_DIM, _GRU_HIDDEN)

        # GRU
        self.gru = nn.GRU(
            input_size=_GRU_INPUT_DIM,
            hidden_size=_GRU_HIDDEN,
            batch_first=True,
        )

        # 离散动作头（共享结构，按 agent_type 设定输出维度）
        n_disc = _LEO_DISC if agent_type == "LEO" else _UAV_DISC
        self.discrete_head = nn.Linear(_CONTEXT_DIM, n_disc)

        # 连续动作头（均值）
        n_cont = _LEO_CONT if agent_type == "LEO" else _UAV_CONT
        self.cont_mean_head = nn.Linear(_CONTEXT_DIM, n_cont)
        self.log_std = nn.Parameter(torch.zeros(n_cont))

        self.n_disc = n_disc
        self.n_cont = n_cont

    def init_hidden(self, graph_enc: torch.Tensor) -> torch.Tensor:
        """
        用全局图编码初始化 GRU 隐藏状态。
        Args:
            graph_enc: (64,)
        Returns:
            h: (1, 1, 64)
        """
        h = F.relu(self.hidden_init(graph_enc))  # (64,)
        return h.unsqueeze(0).unsqueeze(0)       # (1, 1, 64)

    def _attention(self, w_t: torch.Tensor, node_embs: torch.Tensor) -> torch.Tensor:
        """
        点积注意力。
        Args:
            w_t:        (64,) 或 (B, 64)
            node_embs:  (N, 64)
        Returns:
            context: (64,) 或 (B, 64)
        """
        # w_t: (B, 64) after unsqueeze if unbatched
        unbatched = w_t.dim() == 1
        if unbatched:
            w_t = w_t.unsqueeze(0)  # (1, 64)

        # scores: (B, N)
        scores = w_t @ node_embs.T
        alpha = F.softmax(scores, dim=-1)          # (B, N)
        context = alpha @ node_embs                # (B, 64)

        if unbatched:
            context = context.squeeze(0)           # (64,)
        return context

    def forward(
        self,
        h_v_t:      torch.Tensor,                  # (64,)
        L_us_agg:   torch.Tensor,                  # (64,)
        server_agg: torch.Tensor,                  # (64,)
        a_prev:     torch.Tensor,                  # (8,)
        h_prev:     Optional[torch.Tensor],        # (1, 1, 64) or None
        node_embs:  torch.Tensor,                  # (N, 64)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步推断。
        Returns:
            action:   (n_disc + n_cont,)  = (7,) LEO 或 (9,) UAV
            log_prob: scalar
            h_next:   (1, 1, 64)
        """
        # --- GRU ---
        gru_in = torch.cat([h_v_t, L_us_agg, server_agg, a_prev], dim=-1)  # (200,)
        gru_in = gru_in.unsqueeze(0).unsqueeze(0)  # (1, 1, 200)
        gru_out, h_next = self.gru(gru_in, h_prev)  # gru_out: (1, 1, 64)
        w_t = gru_out.squeeze(0).squeeze(0)         # (64,)

        # --- Attention ---
        context = self._attention(w_t, node_embs)  # (64,)
        c_t = torch.cat([w_t, context], dim=-1)    # (128,)

        # --- Discrete head ---
        disc_logits = self.discrete_head(c_t)      # (n_disc,)
        dist_disc = Categorical(logits=disc_logits)
        disc_action = dist_disc.sample()           # scalar
        log_prob_disc = dist_disc.log_prob(disc_action)  # scalar

        # --- Continuous head ---
        cont_mean = torch.sigmoid(self.cont_mean_head(c_t))  # (n_cont,) in [0,1]
        dist_cont = Normal(cont_mean, self.log_std.exp())
        cont_action = dist_cont.sample()           # (n_cont,)
        log_prob_cont = dist_cont.log_prob(cont_action).sum()  # scalar

        log_prob = log_prob_disc + log_prob_cont

        # 拼接动作：[disc_onehot_index(1,)表示为index; cont]
        # 存储格式：[disc_logits(n_disc), cont(n_cont)]，与现有 buffer 兼容逻辑一致
        action = torch.cat([disc_logits.detach(), cont_action.detach()], dim=-1)

        return action, log_prob, h_next

    def evaluate(
        self,
        h_v_t:          torch.Tensor,   # (B, 64)
        L_us_agg:       torch.Tensor,   # (B, 64)
        server_agg:     torch.Tensor,   # (B, 64)
        a_prev:         torch.Tensor,   # (B, 8)
        h_prev:         Optional[torch.Tensor],  # (1, B, 64) or None
        node_embs:      torch.Tensor,   # (N, 64)
        discrete_actions: torch.Tensor, # (B,) int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量评估，用于 PPO 更新。
        Returns:
            log_probs:  (B,)
            entropies:  (B,)
            h_next:     (1, B, 64)
        """
        B = h_v_t.shape[0]
        gru_in = torch.cat([h_v_t, L_us_agg, server_agg, a_prev], dim=-1)  # (B, 200)
        gru_in = gru_in.unsqueeze(1)  # (B, 1, 200)

        gru_out, h_next = self.gru(gru_in, h_prev)  # gru_out: (B, 1, 64)
        w_t = gru_out.squeeze(1)                    # (B, 64)

        context = self._attention(w_t, node_embs)   # (B, 64)
        c_t = torch.cat([w_t, context], dim=-1)     # (B, 128)

        disc_logits = self.discrete_head(c_t)       # (B, n_disc)
        dist_disc = Categorical(logits=disc_logits)
        log_probs = dist_disc.log_prob(discrete_actions.long())  # (B,)
        entropies = dist_disc.entropy()                           # (B,)

        return log_probs, entropies, h_next
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/v2/test_actor_v2.py -v
```

预期：7 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add models/v2/actor_v2.py tests/v2/test_actor_v2.py
git commit -m "feat(v2): implement ActorV2 with GRU+attention and per-agent-type action heads"
```

---

## Task 4：实现 MAPPOAgentV2

**Files:**
- Create: `models/v2/agent_v2.py`
- Create: `tests/v2/test_agent_v2.py`

### 核心设计要点
- `encode()` 一次性缓存 `node_embs / server_embs / graph_enc / topo_order`，并用 `graph_enc` 初始化 `h_pi`
- `act()` 从缓存取 `node_embs[task_id]`，收集上游决策，调用 `ActorV2.forward()`
- Kahn 算法实现拓扑排序（不依赖 networkx，纯 PyTorch/Python）
- Critic 沿用现有 `models.critic.Critic`（不改动）

- [ ] **Step 1: 写失败测试**

`tests/v2/test_agent_v2.py`:

```python
import pytest
import torch
import numpy as np
from models.v2.agent_v2 import MAPPOAgentV2
from utils.config import Config


def _make_graph(n=5):
    dag_x = torch.randn(n, 5)
    # 链式 DAG: 0→1→2→3→4
    edges = [[i, i + 1] for i in range(n - 1)]
    dag_ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    res_x = torch.randn(4, 2)
    res_edges = [[i, j] for i in range(4) for j in range(4) if i != j]
    res_ei = torch.tensor(res_edges, dtype=torch.long).t().contiguous()
    return dag_x, dag_ei, res_x, res_ei


def _make_obs():
    return np.random.randn(37).astype(np.float32)


def test_encode_fills_cache():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(5)

    assert agent.node_embs is None
    agent.encode(dag_x, dag_ei, res_x, res_ei)
    assert agent.node_embs is not None
    assert agent.node_embs.shape == (5, 64)
    assert agent.server_embs.shape == (4, 64)
    assert agent.graph_enc.shape == (64,)
    assert agent.topo_order is not None
    assert len(agent.topo_order) == 5
    assert agent.h_pi is not None   # 用 graph_enc 初始化，不为 None


def test_topo_order_is_valid():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(5)
    agent.encode(dag_x, dag_ei, res_x, res_ei)

    topo = agent.topo_order
    # 链式 DAG：拓扑顺序必须是 [0,1,2,3,4]
    assert topo == list(range(5)), f"Expected [0,1,2,3,4], got {topo}"


def test_act_returns_correct_leo_shape():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(5)
    agent.encode(dag_x, dag_ei, res_x, res_ei)

    obs = _make_obs()
    action, log_prob, h_pi = agent.act(obs)

    assert action.shape == (7,), f"LEO action: {action.shape}"
    assert isinstance(log_prob, float)
    assert h_pi.shape == (1, 1, 64)


def test_act_increments_step_idx():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(5)
    agent.encode(dag_x, dag_ei, res_x, res_ei)

    assert agent.step_idx == 0
    agent.act(_make_obs())
    assert agent.step_idx == 1
    agent.act(_make_obs())
    assert agent.step_idx == 2


def test_reset_episode_clears_cache():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(5)
    agent.encode(dag_x, dag_ei, res_x, res_ei)
    agent.act(_make_obs())

    agent.reset_episode()

    assert agent.node_embs is None
    assert agent.topo_order is None
    assert agent.step_idx == 0
    assert len(agent.decisions) == 0
    assert agent.h_pi is None


def test_uav_act_returns_correct_shape():
    cfg = Config()
    agent = MAPPOAgentV2(agent_id=0, agent_type="UAV", config=cfg)
    dag_x, dag_ei, res_x, res_ei = _make_graph(3)
    agent.encode(dag_x, dag_ei, res_x, res_ei)

    action, log_prob, h_pi = agent.act(_make_obs())
    assert action.shape == (9,), f"UAV action: {action.shape}"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/v2/test_agent_v2.py -v
```

预期：`ImportError: cannot import name 'MAPPOAgentV2'`

- [ ] **Step 3: 实现 `MAPPOAgentV2`**

`models/v2/agent_v2.py`:

```python
import torch
import numpy as np
from typing import Optional, List, Dict, Tuple

from models.v2.gnn_encoder_v2 import GNNEncoderV2
from models.v2.actor_v2 import ActorV2
from models.critic import Critic


def _kahn_topo_sort(n_nodes: int, edge_index: torch.Tensor) -> List[int]:
    """
    Kahn 算法计算拓扑排序。
    Args:
        n_nodes:    节点总数
        edge_index: (2, E) 有向边 [src, dst]
    Returns:
        拓扑排序列表（节点索引）
    """
    in_degree = [0] * n_nodes
    adj: List[List[int]] = [[] for _ in range(n_nodes)]

    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)
            in_degree[d] += 1

    queue = [i for i in range(n_nodes) if in_degree[i] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != n_nodes:
        # DAG 中存在环（不应发生），回退为顺序
        result = list(range(n_nodes))
    return result


class MAPPOAgentV2:
    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        config,
        shared_encoder: Optional[GNNEncoderV2] = None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.device = torch.device(config.device)

        self.encoder = shared_encoder if shared_encoder is not None else GNNEncoderV2().to(self.device)
        self.actor = ActorV2(agent_type=agent_type).to(self.device)
        self.critic = Critic(obs_dim=config.obs_dim, n_agents=config.M).to(self.device)

        # GRU 隐藏状态
        self.h_pi: Optional[torch.Tensor] = None  # (1, 1, 64)
        self.h_V:  Optional[torch.Tensor] = None  # (1, 1, 64)

        # Episode 级缓存
        self.node_embs:   Optional[torch.Tensor] = None  # (N, 64)
        self.server_embs: Optional[torch.Tensor] = None  # (4, 64)
        self.graph_enc:   Optional[torch.Tensor] = None  # (64,)
        self.topo_order:  Optional[List[int]] = None
        self.step_idx:    int = 0
        self.decisions:   Dict[int, torch.Tensor] = {}   # {task_id: action}

    # ------------------------------------------------------------------
    # encode：episode 开始时调用一次
    # ------------------------------------------------------------------

    def encode(
        self,
        dag_x: torch.Tensor,
        dag_edge_index: torch.Tensor,
        res_x: torch.Tensor,
        res_edge_index: torch.Tensor,
    ):
        self.encoder.eval()
        with torch.no_grad():
            dag_x          = dag_x.to(self.device)
            dag_edge_index = dag_edge_index.to(self.device)
            res_x          = res_x.to(self.device)
            res_edge_index = res_edge_index.to(self.device)

            node_embs, server_embs, graph_enc = self.encoder(
                dag_x, dag_edge_index, res_x, res_edge_index
            )

        self.node_embs   = node_embs
        self.server_embs = server_embs
        self.graph_enc   = graph_enc

        n_nodes = dag_x.shape[0]
        self.topo_order = _kahn_topo_sort(n_nodes, dag_edge_index.cpu())

        # 用 graph_enc 初始化 GRU 隐藏状态
        with torch.no_grad():
            self.h_pi = self.actor.init_hidden(graph_enc)  # (1, 1, 64)

        self.step_idx = 0
        self.decisions = {}

    # ------------------------------------------------------------------
    # act：每步调用
    # ------------------------------------------------------------------

    def act(
        self,
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, float, torch.Tensor]:
        assert self.node_embs is not None, "Call encode() before act()"
        assert self.step_idx < len(self.topo_order), "step_idx out of range"

        self.actor.eval()
        with torch.no_grad():
            task_id = self.topo_order[self.step_idx]
            h_v_t = self.node_embs[task_id]                    # (64,)
            server_agg = self.server_embs.mean(dim=0)          # (64,)

            # 收集上游决策
            upstream_actions = [
                self.decisions[t]
                for t in self.topo_order[:self.step_idx]
                if t in self.decisions
            ]
            if upstream_actions:
                stacked = torch.stack(upstream_actions, dim=0)  # (K, d_a)
                # 只取前 64 维用于聚合（填充或截断到 64 维）
                d = stacked.shape[-1]
                if d < 64:
                    pad = torch.zeros(*stacked.shape[:-1], 64 - d, device=self.device)
                    stacked = torch.cat([stacked, pad], dim=-1)
                else:
                    stacked = stacked[..., :64]
                L_us_agg = stacked.mean(dim=0)                 # (64,)
            else:
                L_us_agg = torch.zeros(64, device=self.device)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            a_prev = obs_t[29:37]                              # (8,)

            action, log_prob, h_next = self.actor(
                h_v_t, L_us_agg, server_agg, a_prev, self.h_pi, self.node_embs
            )

        self.decisions[task_id] = action.detach()
        self.step_idx += 1
        self.h_pi = h_next

        return action.cpu().numpy(), log_prob.item(), self.h_pi

    # ------------------------------------------------------------------
    # get_value：每步调用（集中式 Critic）
    # ------------------------------------------------------------------

    def get_value(self, global_obs: np.ndarray) -> Tuple[float, torch.Tensor]:
        self.critic.eval()
        with torch.no_grad():
            g_obs = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device)
            value, h_next = self.critic(g_obs, self.h_V)
            self.h_V = h_next
        return value.item(), self.h_V

    # ------------------------------------------------------------------
    # evaluate_actions：PPO 更新时调用（批量）
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs_batch:        torch.Tensor,   # (B, 37)
        actions_batch:    torch.Tensor,   # (B, d_a)
        global_obs_batch: torch.Tensor,   # (B, 148)
        h_pi_batch:       torch.Tensor,   # (B, 1, 1, 64)
        h_V_batch:        torch.Tensor,   # (B, 1, 1, 64)
        dag_x:            torch.Tensor,
        dag_edge_index:   torch.Tensor,
        res_x:            torch.Tensor,
        res_edge_index:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = obs_batch.shape[0]
        obs_batch        = obs_batch.to(self.device)
        actions_batch    = actions_batch.to(self.device)
        global_obs_batch = global_obs_batch.to(self.device)
        h_pi_batch       = h_pi_batch.to(self.device)
        h_V_batch        = h_V_batch.to(self.device)

        # 编码器前向（评估时重新编码，使用训练模式）
        dag_x          = dag_x.to(self.device)
        dag_edge_index = dag_edge_index.to(self.device)
        res_x          = res_x.to(self.device)
        res_edge_index = res_edge_index.to(self.device)

        self.encoder.train()
        node_embs, server_embs, _ = self.encoder(dag_x, dag_edge_index, res_x, res_edge_index)

        # 从 obs 切片取 a_prev 和上游决策
        a_prev_batch   = obs_batch[:, 29:37]          # (B, 8)
        upstream_batch = obs_batch[:, 5:25]           # (B, 20)

        # L_us_agg: 用 upstream_batch 前 64 维（pad 到 64）
        pad = torch.zeros(B, 44, device=self.device)
        L_us_agg_batch = torch.cat([upstream_batch, pad], dim=-1)[:, :64]  # (B, 64)

        server_agg_batch = server_embs.mean(dim=0).unsqueeze(0).expand(B, -1)  # (B, 64)

        # h_v_t：评估时使用 node_embs[0] 作为代表（与现有 AMAPPOTrainer 的"代表图"做法一致）
        h_v_t_batch = node_embs[0:1].expand(B, -1)   # (B, 64)

        h_pi_init = h_pi_batch.squeeze(1).squeeze(1).unsqueeze(0)  # (1, B, 64)

        discrete_actions = actions_batch[:, :self.actor.n_disc].argmax(dim=-1)  # (B,)

        self.actor.train()
        log_probs, entropies, h_pi_next = self.actor.evaluate(
            h_v_t_batch, L_us_agg_batch, server_agg_batch,
            a_prev_batch, h_pi_init, node_embs, discrete_actions,
        )

        h_V_init = h_V_batch.squeeze(1).squeeze(1).unsqueeze(0)  # (1, B, 64)
        self.critic.train()
        values, _ = self.critic(global_obs_batch, h_V_init)
        values = values.squeeze(-1)

        h_pi_new = h_pi_next.squeeze(0).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 64)
        return log_probs, entropies, values, h_pi_new

    # ------------------------------------------------------------------
    # reset_episode
    # ------------------------------------------------------------------

    def reset_episode(self):
        self.node_embs   = None
        self.server_embs = None
        self.graph_enc   = None
        self.topo_order  = None
        self.step_idx    = 0
        self.decisions   = {}
        self.h_pi        = None
        self.h_V         = None

    def parameters(self):
        return (
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/v2/test_agent_v2.py -v
```

预期：6 个测试全部 PASS

- [ ] **Step 5: 运行全部 v2 测试**

```bash
python -m pytest tests/v2/ -v
```

预期：全部 PASS（Task 2 + Task 3 + Task 4 的测试）

- [ ] **Step 6: Commit**

```bash
git add models/v2/agent_v2.py tests/v2/test_agent_v2.py
git commit -m "feat(v2): implement MAPPOAgentV2 with encode/act/reset_episode interface"
```

---

## Task 5：实现 AMAPPOv2Trainer

**Files:**
- Create: `algorithms/amappo_v2.py`

### 核心设计要点
- 直接复制 `AMAPPOTrainer` 的结构，仅替换 Agent/Encoder 类型，并在 `_run_episode()` 开头插入 `agent.encode()` 调用
- `_build_graph_inputs_v2()` 复用 `algorithms.mappo._build_graph_inputs` 的逻辑，但不 import 原 Trainer
- `evaluate_actions` 传入 `dag_x/ei/res_x/ei` 作为代表图（与原 AMAPPOTrainer 行为一致）

- [ ] **Step 1: 写 `algorithms/amappo_v2.py`**

```python
"""
AMAPPOv2 Trainer — 严格对齐论文编码器-解码器架构的异步 MAPPO 实现。

与 AMAPPOTrainer 的唯一结构差异：
  - 使用 GNNEncoderV2（返回节点级嵌入）
  - 使用 MAPPOAgentV2（encode-once + step-level decode）
  - _run_episode() 开头调用 agent.encode()
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from utils.config import Config
from utils.buffer import AgentBuffer, GlobalBuffer, Transition
from utils.logger import Logger
from models.v2.gnn_encoder_v2 import GNNEncoderV2
from models.v2.agent_v2 import MAPPOAgentV2
from env.sec_env import SECEnv


def _build_graph_inputs_v2(env: SECEnv, agent_id: int):
    """
    从环境构造 DAG 和资源图张量（与 mappo._build_graph_inputs 相同逻辑，独立实现）。
    """
    import torch
    dag = env.dags[agent_id]
    nodes_sorted = sorted(dag.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_sorted)}

    features = []
    for n in nodes_sorted:
        d = dag.nodes[n]
        features.append([
            float(d.get("D_in", 0.0)),
            float(d.get("D_out", 0.0)),
            float(d.get("C", 0.0)),
            float(d.get("deadline_rem", 0.0)),
            float(d.get("topo_pos", 0.0)),
        ])
    dag_x = torch.tensor(features, dtype=torch.float32)

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in dag.edges()]
    if edges:
        dag_ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        dag_ei = torch.zeros((2, 0), dtype=torch.long)

    # Resource graph: 4 nodes
    res_features = []
    for srv in ["local", "uav", "sat", "cloud"]:
        info = env.resource_graph.get(srv, {})
        res_features.append([
            float(info.get("load", 0.0)),
            float(info.get("capacity", 1.0)),
        ])
    res_x = torch.tensor(res_features, dtype=torch.float32)

    res_nodes = 4
    res_edges = [[i, j] for i in range(res_nodes) for j in range(res_nodes) if i != j]
    res_ei = torch.tensor(res_edges, dtype=torch.long).t().contiguous()

    return dag_x, dag_ei, res_x, res_ei


class AMAPPOv2Trainer:
    """AMAPPOv2：节点级嵌入 + GRU + 注意力的异步 MAPPO 训练器。"""

    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)

        # 共享 GNNEncoderV2
        self.shared_encoder = GNNEncoderV2().to(self.device)

        # agent_types 从 config 获取，不存在时默认全 LEO
        agent_types = getattr(config, "agent_types", ["LEO"] * config.M)

        self.agents: List[MAPPOAgentV2] = [
            MAPPOAgentV2(
                agent_id=m,
                agent_type=agent_types[m],
                config=config,
                shared_encoder=self.shared_encoder,
            )
            for m in range(config.M)
        ]

        # 统一优化器
        all_params = list(self.shared_encoder.parameters())
        for agent in self.agents:
            all_params += list(agent.actor.parameters())
            all_params += list(agent.critic.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=config.lr)

        # Buffers
        self.agent_buffers: List[AgentBuffer] = [
            AgentBuffer(agent_id=m) for m in range(config.M)
        ]
        self.global_buffer = GlobalBuffer(capacity=50000)

        # 环境与日志
        self.env = SECEnv(config)
        os.makedirs(config.log_dir, exist_ok=True)
        self.logger = Logger(log_dir=config.log_dir, algo_name="amappo_v2")
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self._train_step = 0
        self._agent_decision_count: np.ndarray = np.zeros(config.M, dtype=int)

    # ------------------------------------------------------------------

    def train(self):
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
                    f"[AMAPPOv2] ep={ep:4d}  reward={ep_reward:8.3f}  "
                    f"T={ep_info.get('T_total', 0):.4f}  "
                    f"E={ep_info.get('E_total', 0):.4e}  "
                    f"decisions={decisions_per_agent}"
                )
                self._agent_decision_count[:] = 0

            if ep % self.cfg.save_interval == 0:
                self._save_checkpoint(ep)

        self.logger.close()
        print("[AMAPPOv2] Training complete.")

    # ------------------------------------------------------------------

    def _run_episode(self) -> Tuple[float, dict]:
        obs_dict = self.env.reset()

        for agent in self.agents:
            agent.reset_episode()
        for buf in self.agent_buffers:
            buf.clear()

        # ← 与原 AMAPPOTrainer 的唯一结构差异：encode 一次
        for m, agent in enumerate(self.agents):
            dag_x, dag_ei, res_x, res_ei = _build_graph_inputs_v2(self.env, m)
            agent.encode(dag_x, dag_ei, res_x, res_ei)

        local_clocks: np.ndarray = np.zeros(self.cfg.M, dtype=int)
        total_reward = 0.0
        last_info: dict = {}
        global_step = 0
        last_obs: Dict[int, np.ndarray] = {m: obs_dict[m] for m in range(self.cfg.M)}
        action_dict: Dict[int, np.ndarray] = {
            m: np.zeros(self.cfg.action_dim) for m in range(self.cfg.M)
        }

        def _global_obs() -> np.ndarray:
            return np.concatenate([last_obs[m] for m in range(self.cfg.M)], axis=0)

        while True:
            global_obs = _global_obs()
            available_agents = [m for m in range(self.cfg.M) if global_step >= local_clocks[m]]

            for m in available_agents:
                agent = self.agents[m]
                obs_m = last_obs[m]

                action, log_prob, h_pi = agent.act(obs_m)
                value, h_V = agent.get_value(global_obs)
                action_dict[m] = action

                task_node = self.env._current_task(m)
                if task_node is not None:
                    c_cycles = self.env.dags[m].nodes[task_node].get("C", 1.0)
                    mean_slots = max(1, int(c_cycles / 0.5))
                    exec_slots = max(1, np.random.geometric(1.0 / mean_slots))
                else:
                    exec_slots = 1

                local_clocks[m] = global_step + exec_slots
                self._agent_decision_count[m] += 1

                t = Transition(
                    obs=obs_m,
                    action=action,
                    reward=0.0,
                    h_pi=h_pi.detach().cpu().numpy(),
                    h_V=h_V.detach().cpu().numpy(),
                    global_obs=global_obs.copy(),
                    done=False,
                )
                self.agent_buffers[m].add(t)

            next_obs_dict, rew_dict, done, info = self.env.step(action_dict)
            last_info = info

            for m in available_agents:
                reward = rew_dict.get(m, 0.0)
                total_reward += reward
                buf = self.agent_buffers[m]
                if len(buf.transitions) > 0:
                    buf.transitions[-1].reward = reward
                    buf.transitions[-1].done   = done

            for m in range(self.cfg.M):
                last_obs[m] = next_obs_dict[m]

            global_step += 1
            if done:
                break

        for buf in self.agent_buffers:
            self.global_buffer.add_from_agent_buffer(buf)
            buf.clear()

        avg_reward = total_reward / max(1, sum(self._agent_decision_count))
        return avg_reward, last_info

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
        dag_x, dag_ei, res_x, res_ei = _build_graph_inputs_v2(self.env, 0)
        dag_x  = dag_x.to(self.device)
        dag_ei = dag_ei.to(self.device)
        res_x  = res_x.to(self.device)
        res_ei = res_ei.to(self.device)

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
        self.shared_encoder.train()
        agent.actor.train()
        agent.critic.train()

        log_probs_new, entropies, values_new, _ = agent.evaluate_actions(
            obs_t, actions_t, global_obs_t, h_pi_t, h_V_t,
            dag_x, dag_ei, res_x, res_ei,
        )

        log_probs_old = log_probs_new.detach()
        ratio  = torch.exp(log_probs_new - log_probs_old)
        surr1  = ratio * advantages_t
        surr2  = torch.clamp(ratio, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages_t
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

    def _save_checkpoint(self, episode: int):
        path = os.path.join(self.cfg.checkpoint_dir, f"amappo_v2_ep{episode}.pt")
        state = {
            "episode": episode,
            "shared_encoder": self.shared_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        for m, agent in enumerate(self.agents):
            state[f"actor_{m}"]  = agent.actor.state_dict()
            state[f"critic_{m}"] = agent.critic.state_dict()
        torch.save(state, path)
        print(f"[AMAPPOv2] Checkpoint saved: {path}")
```

- [ ] **Step 2: 验证语法**

```bash
python -c "from algorithms.amappo_v2 import AMAPPOv2Trainer; print('OK')"
```

预期：`OK`

- [ ] **Step 3: 验证原有 Trainer 未受影响**

```bash
python -c "from algorithms.amappo import AMAPPOTrainer; print('OK')"
python -c "from algorithms.mappo import MAPPOTrainer; print('OK')"
```

预期：两行均输出 `OK`

- [ ] **Step 4: Commit**

```bash
git add algorithms/amappo_v2.py
git commit -m "feat(v2): implement AMAPPOv2Trainer with encode-once episode structure"
```

---

## Task 6：实现 train_v2.py 入口

**Files:**
- Create: `experiments/train_v2.py`

- [ ] **Step 1: 写 `experiments/train_v2.py`**

```python
"""
AMAPPOv2 独立训练入口。

用法：
    python experiments/train_v2.py --epochs 1500 --seed 42 --device cuda

此脚本仅 import v2 模块路径，与原有 train.py 完全隔离。
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from algorithms.amappo_v2 import AMAPPOv2Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train AMAPPOv2")
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--seed",            type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--gamma",           type=float, default=None)
    parser.add_argument("--gae_lambda",      type=float, default=None)
    parser.add_argument("--eps_clip",        type=float, default=None)
    parser.add_argument("--mini_batch_size", type=int,   default=None)
    parser.add_argument("--max_grad_norm",   type=float, default=None)
    parser.add_argument("--gru_hidden",      type=int,   default=None)
    parser.add_argument("--log_interval",    type=int,   default=None)
    parser.add_argument("--save_interval",   type=int,   default=None)
    parser.add_argument("--log_dir",         type=str,   default=None)
    parser.add_argument("--checkpoint_dir",  type=str,   default=None)
    parser.add_argument("--device",          type=str,   default=None, choices=["cpu", "cuda"])
    parser.add_argument("--max_steps",       type=int,   default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = Config()
    cfg.algo = "amappo_v2"

    override_fields = [
        "epochs", "seed", "lr", "gamma", "gae_lambda", "eps_clip",
        "mini_batch_size", "max_grad_norm", "gru_hidden", "log_interval",
        "save_interval", "log_dir", "checkpoint_dir", "device", "max_steps",
    ]
    for field in override_fields:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)

    set_seed(cfg.seed)

    print(f"[train_v2.py] algo=amappo_v2  epochs={cfg.epochs}  seed={cfg.seed}")
    print(f"              device={cfg.device}  lr={cfg.lr}  mini_batch={cfg.mini_batch_size}")

    start_time = time.time()
    trainer = AMAPPOv2Trainer(cfg)
    trainer.train()

    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"[train_v2.py] Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证导入**

```bash
python -c "
import sys, os
sys.path.insert(0, '.')
from experiments.train_v2 import main
print('import OK')
"
```

预期：`import OK`

- [ ] **Step 3: 验证原有 train.py 未受影响**

```bash
python -c "
import sys, os
sys.path.insert(0, '.')
from experiments.train import main
print('import OK')
"
```

预期：`import OK`

- [ ] **Step 4: Commit**

```bash
git add experiments/train_v2.py
git commit -m "feat(v2): add independent train_v2.py entry point"
```

---

## Task 7：整合验证

- [ ] **Step 1: 运行全部 v2 单元测试**

```bash
python -m pytest tests/v2/ -v
```

预期：全部 PASS（Task 2 + Task 3 + Task 4 共计 17 个测试）

- [ ] **Step 2: 验证 v2 子包完整导入**

```bash
python -c "
from models.v2 import GNNEncoderV2, ActorV2, MAPPOAgentV2
from algorithms.amappo_v2 import AMAPPOv2Trainer
print('All v2 imports OK')
"
```

预期：`All v2 imports OK`

- [ ] **Step 3: 验证原有 AMAPPO 完全未受影响**

```bash
python -c "
from models.gnn_encoder import GNNEncoder
from models.actor import Actor
from models.agent import MAPPOAgent
from algorithms.amappo import AMAPPOTrainer
from algorithms.mappo import MAPPOTrainer
print('All original imports OK')
"
```

预期：`All original imports OK`

- [ ] **Step 4: 快速 smoke test（不运行完整训练）**

```python
# 保存为 tests/v2/test_smoke.py 并运行
import torch, numpy as np
from utils.config import Config
from models.v2.gnn_encoder_v2 import GNNEncoderV2
from models.v2.actor_v2 import ActorV2
from models.v2.agent_v2 import MAPPOAgentV2


def test_full_encode_act_cycle():
    cfg = Config()
    cfg.device = "cpu"
    encoder = GNNEncoderV2()

    agent = MAPPOAgentV2(agent_id=0, agent_type="LEO", config=cfg, shared_encoder=encoder)

    n = 5
    dag_x = torch.randn(n, 5)
    edges = [[i, i + 1] for i in range(n - 1)]
    dag_ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    res_x = torch.randn(4, 2)
    res_edges = [[i, j] for i in range(4) for j in range(4) if i != j]
    res_ei = torch.tensor(res_edges, dtype=torch.long).t().contiguous()

    agent.encode(dag_x, dag_ei, res_x, res_ei)

    for step in range(n):
        obs = np.random.randn(37).astype(np.float32)
        action, log_prob, h_pi = agent.act(obs)
        assert action.shape == (7,), f"step {step}: {action.shape}"

    agent.reset_episode()
    assert agent.step_idx == 0
```

```bash
python -m pytest tests/v2/test_smoke.py -v
```

预期：PASS

- [ ] **Step 5: 最终 Commit**

```bash
git add tests/v2/test_smoke.py
git commit -m "test(v2): add smoke test for full encode-act cycle"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] 独立子包 `models/v2/` — Task 1
- [x] `GNNEncoderV2` 双向拼接 + 节点级输出 + GraphEnc FC — Task 2
- [x] `ActorV2` GRU(200→64) + 注意力(128维上下文) + LEO/UAV 动作头 — Task 3
- [x] `MAPPOAgentV2` encode/act/reset/evaluate_actions — Task 4
- [x] `AMAPPOv2Trainer` episode 开头 encode() 调用 — Task 5
- [x] `experiments/train_v2.py` 独立入口 — Task 6
- [x] 原有文件零改动验证 — Task 7

**类型一致性:**
- `GNNEncoderV2.forward()` 返回 `(node_embs, server_embs, graph_enc)` — `agent_v2.py` 中 `encode()` 解包时顺序一致 ✓
- `ActorV2.forward()` 返回 `(action, log_prob, h_next)` — `agent_v2.py` 中 `act()` 解包顺序一致 ✓
- `ActorV2.evaluate()` 返回 `(log_probs, entropies, h_next)` — `agent_v2.py` 中 `evaluate_actions()` 解包顺序一致 ✓
- `_kahn_topo_sort` 返回 `List[int]` — `act()` 中 `topo_order[step_idx]` 直接索引 ✓
