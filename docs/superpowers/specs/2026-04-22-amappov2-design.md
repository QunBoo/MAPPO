# AMAPPOv2 设计文档

**日期**：2026-04-22  
**目标**：在不破坏现有 AMAPPO 算法运行的基础上，新开发 AMAPPOv2 算法，严格对齐论文中的编码器-解码器架构。  
**参考**：[差异分析](../../specs/2026-04-21-diff.md)、[编码器设计](../../论文编码器encoder设计.md)、[解码器设计](../../论文解码器decoder设计.md)

---

## 一、设计决策

| 问题 | 决策 |
|------|------|
| 隔离策略 | 独立子包 `models/v2/`，原有文件零改动 |
| 拓扑排序来源 | Agent 自行计算（Kahn 算法，从 DAG 边索引推导） |
| 服务器图特征 | 暂用现有 2 维 `[cpu_freq_norm, current_load]`，对齐留待后续迭代 |
| 动作空间 | 按 `agent_type`（LEO / UAV）区分，共享 GRU + 注意力层，独立输出头 |
| 训练入口 | 独立脚本 `experiments/train_v2.py`，不修改现有 `train.py` |
| 解码方式 | **Encode-once + Step-level Decode**：episode 开始时 `encode()` 一次，每步 `act()` 使用缓存嵌入 |

---

## 二、包结构

```
AMAPPO复现/
├── models/
│   ├── gnn_encoder.py          ← 原有，不改动
│   ├── actor.py                ← 原有，不改动
│   ├── agent.py                ← 原有，不改动
│   └── v2/                     ← 新建子包
│       ├── __init__.py
│       ├── gnn_encoder_v2.py   # GNNEncoderV2
│       ├── actor_v2.py         # ActorV2
│       └── agent_v2.py         # MAPPOAgentV2
│
├── algorithms/
│   ├── amappo.py               ← 原有，不改动
│   └── amappo_v2.py            ← 新建：AMAPPOv2Trainer
│
└── experiments/
    ├── train.py                ← 原有，不改动
    └── train_v2.py             ← 新建：独立训练入口
```

**组件职责边界**：

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `GNNEncoderV2` | 图编码，返回节点级嵌入 | `dag_x, dag_edge_index, res_x, res_edge_index` | `node_embs (N,64)`, `server_embs (4,64)`, `graph_enc (64,)` |
| `ActorV2` | 单步 GRU + 注意力 + 动作头 | `h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs, agent_type` | `action, log_prob, h_next` |
| `MAPPOAgentV2` | 缓存管理 + 接口封装 | DAG 图数据 / 观测向量 | 动作、log_prob、value |
| `AMAPPOv2Trainer` | 训练循环，插入 `encode()` 调用 | config | 训练结果 |

---

## 三、GNNEncoderV2

### 3.1 任务 DAG 编码器（双向拼接）

**关键改动**：融合方式从求和改为拼接，保留方向性信息。每层每路输出 32 维，拼接后恢复 64 维。

```
Layer 1:
  h_fwd1 = SAGEConv_fwd(x, edge_index)           # (N, 32)  下游方向
  h_bwd1 = SAGEConv_bwd(x, edge_index_rev)       # (N, 32)  上游方向
  h1 = ReLU(BN(concat([h_fwd1, h_bwd1])))        # (N, 64)  拼接融合

Layer 2:
  h_fwd2 = SAGEConv_fwd(h1, edge_index)          # (N, 32)
  h_bwd2 = SAGEConv_bwd(h1, edge_index_rev)      # (N, 32)
  node_embs = ReLU(BN(concat([h_fwd2, h_bwd2]))) # (N, 64)  任务节点嵌入
```

参数：`in_dim=5`（与现有 DAG 特征对齐），`hidden_dim=64`，2 层。

### 3.2 服务器图编码器

与现有 `ResourceGraphEncoder` 相同，`in_dim=2`，输出 `server_embs: (4, 64)`。

### 3.3 全局图编码

在 max-pool 前增加 FC 投影层（论文要求，当前实现缺失）：

```
graph_enc = max_pool(FC(node_embs))   # FC: (64,) → (64,)，输出 (64,)
```

### 3.4 输出接口

```python
def forward(
    dag_x: Tensor,           # (N, 5)
    dag_edge_index: Tensor,  # (2, E)
    res_x: Tensor,           # (4, 2)
    res_edge_index: Tensor,  # (2, E_r)
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns:
        node_embs:   (N, 64)   任务节点嵌入，解码器注意力的 key/value
        server_embs: (4, 64)   服务器节点嵌入，GRU 输入
        graph_enc:   (64,)     全局图编码，GRU 初始化
    """
```

---

## 四、ActorV2（单步解码器）

每次 `act()` 调用执行三阶段计算。

### 4.1 阶段 1 — GRU 更新

```
gru_input = concat([
    h_{v_t},     # (64,)  当前任务节点嵌入（来自缓存的 node_embs[task_id]）
    L_us_agg,    # (64,)  上游决策均值池化（无上游时为零向量）
    server_agg,  # (64,)  server_embs.mean(dim=0)
    a_prev,      # (8,)   上一步动作（沿用现有 buffer 格式，LEO/UAV 共用 8 维）
])               # 总：64+64+64+8 = 200 维

w_t, h_next = GRU(gru_input, w_{t-1})   # w_t: (64,)
```

GRU 隐藏维度：64。输入维度：200（LEO 和 UAV 共用，动作头分叉仅在输出侧）。

**GRU 初始化**（episode 开始时，由 `MAPPOAgentV2.encode()` 调用）：

```
w_0 = Linear(64 → 64)(graph_enc)   # 用全局图编码投影初始化，替代零初始化
```

### 4.2 阶段 2 — 注意力（全局感知）

```
scores  = w_t @ node_embs.T         # (N,)   点积注意力分数
alpha   = softmax(scores)            # (N,)
context = alpha @ node_embs          # (64,)  加权全局任务信息

c_t = concat([w_t, context])         # (128,) 上下文向量 → 输入 Actor 头
```

### 4.3 阶段 3 — 动作头（按 agent_type 区分）

共享 GRU + 注意力层，最终输出头按类型分叉：

| agent_type | 离散动作 | 连续动作 | 动作总维度 `d_a` |
|-----------|---------|---------|----------------|
| `LEO` | Categorical(4)：卸载目标（本地/UAV/卫星/云） | 3 维：带宽 z、功率 P、计算资源 f | **7** |
| `UAV` | Categorical(4)：卸载目标 | 5 维：z、P、f_m、f_k、轨迹 q | **9** |

离散动作通过 `Categorical` 分布采样，连续动作通过 `Normal` 分布采样（均值 + log_std）。

### 4.4 接口

```python
def forward(
    h_v_t:      Tensor,   # (64,)   当前任务节点嵌入
    L_us_agg:   Tensor,   # (64,)   上游决策聚合
    server_agg: Tensor,   # (64,)   服务器嵌入聚合
    a_prev:     Tensor,   # (8,)    上一步动作（沿用现有 buffer 格式，8 维）
    h_prev:     Tensor,   # (1,1,64) GRU 隐藏状态
    node_embs:  Tensor,   # (N, 64) 全部任务节点嵌入（注意力用）
    agent_type: str,      # 'LEO' or 'UAV'
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns: action (d_a,), log_prob (scalar), h_next (1,1,64)"""

def evaluate(
    ...                   # 批量版本，用于 PPO 更新
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns: log_probs (B,), entropies (B,), h_next (1,B,64)"""
```

---

## 五、MAPPOAgentV2 状态管理

### 5.1 持有的状态

```python
# 跨 step 持久（GRU 隐藏状态）
h_pi: Tensor | None    # (1, 1, 64)  Actor GRU 隐藏状态
h_V:  Tensor | None    # (1, 1, 64)  Critic GRU 隐藏状态

# Episode 级缓存（encode() 后填充，reset_episode() 清空）
node_embs:   Tensor | None   # (N, 64)  任务节点嵌入
server_embs: Tensor | None   # (4, 64)  服务器节点嵌入
graph_enc:   Tensor | None   # (64,)    全局图编码
topo_order:  list | None     # 拓扑排序节点索引列表（Kahn 算法）
step_idx:    int             # 当前拓扑步骤计数器，初始为 0
decisions:   dict            # {task_id: action_tensor}  已决策记录
```

### 5.2 三个关键方法

**`encode(dag_x, dag_edge_index, res_x, res_edge_index)`**  
episode 开始时由 Trainer 调用一次：
1. 运行 `GNNEncoderV2`，填充 `node_embs / server_embs / graph_enc` 缓存
2. 用 Kahn 算法从 `dag_edge_index` 计算 `topo_order`
3. 用 `graph_enc` 初始化 `h_pi`（通过 ActorV2 的投影层）
4. 重置 `step_idx = 0`，清空 `decisions`

**`act(obs) -> (action, log_prob, h_pi)`**  
每步由 Runner 调用：
1. `task_id = topo_order[step_idx]`
2. `h_v_t = node_embs[task_id]`
3. 从 `decisions` 中收集上游决策，均值池化得 `L_us_agg`
4. `server_agg = server_embs.mean(dim=0)`
5. 从 `obs` 切片取 `a_prev`（上一步动作）
6. 调用 `ActorV2.forward(...)`，得到 `action, log_prob, h_next`
7. `decisions[task_id] = action`；`step_idx += 1`；更新 `h_pi`

**`reset_episode()`**  
episode 结束时调用，清空所有 episode 级缓存，`h_pi / h_V` 置 None。

### 5.3 观测切片

沿用现有 37 维观测布局，仅取 `a_prev = obs[29:37]`（8 维），动作维度在 `ActorV2` 内部按 `agent_type` 截断使用。

---

## 六、AMAPPOv2Trainer

复用现有 `AMAPPOTrainer` 的训练循环骨架，最小化差异：

```python
class AMAPPOv2Trainer:
    def __init__(self, config):
        encoder = GNNEncoderV2().to(device)       # 所有 agent 共享编码器
        self.agents = [
            MAPPOAgentV2(
                agent_id=i,
                agent_type=config.agent_types[i],  # 'LEO' or 'UAV'
                shared_encoder=encoder,
                config=config,
            )
            for i in range(config.M)
        ]

    def collect_episode(self, env):
        # env.get_graph_data() 为 AMAPPOv2Trainer 新增的约定调用：
        # 从环境获取当前 episode 的 DAG 图数据和资源图数据。
        # 实现上等价于现有 AMAPPOTrainer 中已有的图数据获取逻辑，
        # 提取为独立方法以明确 encode/decode 分离边界。
        dag_x, dag_ei, res_x, res_ei = env.get_graph_data()

        # ← 与原 AMAPPOTrainer 的唯一结构差异
        for agent in self.agents:
            agent.encode(dag_x, dag_ei, res_x, res_ei)

        # 以下 step 循环、buffer 存储、PPO 更新与原 Trainer 相同
        ...
```

`Config` 新增字段 `agent_types: list[str]`，默认 `['LEO'] * M`，不影响现有 Config 用法。

---

## 七、train_v2.py 入口

```
python experiments/train_v2.py --epochs 1500 --seed 42 --device cuda
```

只 import v2 模块路径，不引用任何 `models/gnn_encoder.py`、`models/actor.py`、`algorithms/amappo.py` 等原有路径，确保完全隔离。

---

## 八、关键维度汇总

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 任务节点嵌入维度 | 64 | `node_embs` 每行 |
| 服务器节点嵌入维度 | 64 | `server_embs` 每行 |
| 全局图编码维度 | 64 | `graph_enc`，与 GRU 隐藏维度对齐 |
| GRU 隐藏维度 | 64 | `h_pi` |
| 上下文向量维度 | 128 | `concat([w_t, context])` |
| GRU 输入维度 | 200 | `64+64+64+8`（LEO/UAV 共用，输出头分叉） |
| DAG 节点特征维度 | 5 | 沿用现有 `in_dim=5` |
| 服务器节点特征维度 | 2 | 暂用现有 `[cpu_freq_norm, current_load]` |

---

## 九、与现有实现的差异对照

| 差异项 | 当前 AMAPPO | AMAPPOv2 |
|--------|------------|-----------|
| 编码器输出粒度 | 128 维图级嵌入（标量） | `(N,64)` 节点嵌入 + `(4,64)` 服务器嵌入 + `(64,)` 全局编码 |
| 双向融合方式 | `h_fwd + h_bwd`（求和） | `concat([h_fwd, h_bwd])`（拼接） |
| 全局图编码 | 直接 max-pool | FC 投影后 max-pool |
| GRU 初始化 | 零初始化 | `Linear(graph_enc)` 投影 |
| GRU 输入 | 图级嵌入 + 上游决策 + 服务器状态 + 上一动作 | **当前任务节点嵌入** + 上游决策 + 服务器嵌入聚合 + 上一动作 |
| 注意力机制 | 无 | GRU 输出后：点积注意力 → softmax → 加权上下文向量 |
| 动作空间 | 统一 8 维（4+4） | LEO: 7 维（4+3），UAV: 9 维（4+5） |
| 服务器嵌入来源 | obs 切片（原始观测） | GNNEncoderV2 输出（GNN 编码后） |
| encode/decode 分离 | 无（每步重跑编码器） | episode 开始 `encode()` 一次，步骤内复用缓存 |
