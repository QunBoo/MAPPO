# AMAPPOv2 算法技术规格文档

> **版本**: v2.2  
> **最后更新**: 2026-04-23  
> **核心模块**: `algorithms/amappo_v2.py`, `models/v2/`, `utils/buffer.py`, `env/sec_env.py`

---

## 1. 概述

AMAPPOv2 (Asynchronous Multi-Agent Proximal Policy Optimization v2) 是一种面向卫星边缘计算 (Satellite Edge Computing, SEC) 场景的多智能体强化学习算法。它基于 MAPPO 框架，引入了**异步决策机制**、**编码-解码分离架构** (Encode-Once Decode-Many) 和**节点级注意力机制**，以更高效地处理 DAG 任务调度的时序依赖和异构智能体协作问题。

### 1.1 与 AMAPPOv1 的核心区别

| 特性 | AMAPPOv1 | AMAPPOv2 |
|------|----------|----------|
| GNN 编码器 | `GNNEncoder` (返回图级编码) | `GNNEncoderV2` (返回节点级嵌入) |
| 策略网络 | 无 GRU / 无注意力 | `ActorV2` (GRU + 点积注意力) |
| 编码策略 | 每步重新编码 | Episode 开始时编码一次 (`encode`)，每步仅解码 (`act`) |
| 智能体类型 | 统一 LEO | 支持 LEO / UAV 异构动作空间 |
| 隐状态初始化 | 零初始化 | 从图级编码 `graph_enc` 投影初始化 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                     AMAPPOv2Trainer                              │
│                                                                  │
│  ┌─────────────┐    ┌──────────────────────────────────────┐     │
│  │ Shared      │    │  MAPPOAgentV2 × M                    │     │
│  │ GNNEncoderV2│───▶│  ┌─────────┐  ┌─────────┐           │     │
│  │             │    │  │ ActorV2 │  │ Critic  │           │     │
│  └─────────────┘    │  │ (GRU+   │  │ (GRU)   │           │     │
│                     │  │  Attn)  │  │         │           │     │
│  ┌─────────────┐    │  └─────────┘  └─────────┘           │     │
│  │ GlobalBuffer│◀───│       │              │              │     │
│  │ AgentBuffer │    │  encode() → act() → get_value()     │     │
│  │ × M         │    │                                      │     │
│  └─────────────┘    └──────────────────────────────────────┘     │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐                             │
│  │   SECEnv    │    │   Optimizer │                             │
│  │ (4-tier SEC)│    │   (Adam)    │                             │
│  └─────────────┘    └─────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
AMAPPOv2Trainer
├── GNNEncoderV2 (共享)
│   ├── TaskDAGEncoderV2
│   │   └── _BidirectionalSAGELayer × 2
│   └── ResourceGraphEncoderV2
│       └── SAGEConv × 2
├── MAPPOAgentV2 × M
│   ├── ActorV2
│   │   ├── GRU (input=200, hidden=64)
│   │   ├── Dot-Product Attention
│   │   ├── Discrete Head (Linear → Categorical)
│   │   └── Continuous Head (Linear → Normal)
│   └── Critic
│       ├── Linear(148 → 128)
│       ├── GRU(128 → 64)
│       └── Linear(64 → 1)
├── AgentBuffer × M
├── GlobalBuffer
├── SECEnv
└── Logger
```

---

## 3. 核心组件详解

### 3.1 GNNEncoderV2 — 双图编码器

**文件**: `models/v2/gnn_encoder_v2.py`

GNNEncoderV2 接收两个图结构输入，分别编码后输出三种表示：

#### 输入

| 输入 | 形状 | 说明 |
|------|------|------|
| `dag_x` | `(N, 5)` | DAG 任务节点特征：`[D_in, D_out, C, deadline_rem, topo_pos]` |
| `dag_edge_index` | `(2, E)` | DAG 边索引（有向） |
| `res_x` | `(4, 2)` | 资源节点特征：`[load, capacity]`（local, UAV, sat, cloud） |
| `res_edge_index` | `(2, E_res)` | 资源图边索引（全连接） |

#### 输出

| 输出 | 形状 | 说明 |
|------|------|------|
| `node_embs` | `(N, 64)` | DAG 每个任务节点的嵌入向量 |
| `server_embs` | `(4, 64)` | 4 个计算服务器的嵌入向量 |
| `graph_enc` | `(64,)` | 全局图编码（max-pooling + 投影） |

#### TaskDAGEncoderV2

采用 **双向 GraphSAGE** 架构，分别沿正向边和反向边传播信息，拼接后经 BatchNorm + ReLU：

```
输入 (N, 5)
  → BidirectionalSAGELayer(5→64): [SAGE_fwd(5→32) ‖ SAGE_bwd(5→32)] → BN → ReLU
  → BidirectionalSAGELayer(64→64): [SAGE_fwd(64→32) ‖ SAGE_bwd(64→32)] → BN → ReLU
  → node_embs (N, 64)
  → graph_enc_proj(node_embs) → ReLU → MaxPool → graph_enc (64,)
```

#### ResourceGraphEncoderV2

采用标准 GraphSAGE 两层卷积：

```
输入 (4, 2) → SAGEConv(2→64) → BN → ReLU → SAGEConv(64→64) → BN → ReLU → server_embs (4, 64)
```

### 3.2 ActorV2 — GRU + 注意力策略网络

**文件**: `models/v2/actor_v2.py`

ActorV2 是 AMAPPOv2 的核心创新，支持 **LEO** 和 **UAV** 两种异构动作空间。

#### 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| `GRU_INPUT_DIM` | 200 | `[h_v_t(64) ‖ L_us_agg(64) ‖ server_agg(64) ‖ a_prev(8)]` |
| `GRU_HIDDEN` | 64 | GRU 隐状态维度 |
| `NODE_EMB_DIM` | 64 | 节点嵌入维度 |
| `CONTEXT_DIM` | 128 | `[w_t(64) ‖ context(64)]` |

#### 异构动作空间

| 智能体类型 | 离散动作维度 | 连续动作维度 | 连续动作含义 |
|-----------|-------------|-------------|-------------|
| LEO | 4 | 3 | z, P, f |
| UAV | 4 | 5 | z, P, f_m, f_k, q |

#### 前向流程 (`forward`)

```
输入: h_v_t(64), L_us_agg(64), server_agg(64), a_prev(8), h_prev(1,1,64), node_embs(N,64)

1. GRU:
   gru_in = [h_v_t ‖ L_us_agg ‖ server_agg ‖ a_prev]  → (200,)
   gru_in.reshape → (1, 1, 200)
   gru_out, h_next = GRU(gru_in, h_prev)               → w_t (64,)

2. Attention:
   scores = w_t @ node_embs.T                          → (N,)
   alpha  = softmax(scores)                            → (N,)
   context = alpha @ node_embs                         → (64,)
   c_t = [w_t ‖ context]                               → (128,)

3. Discrete Head:
   disc_logits = Linear(c_t)                           → (n_disc,)
   dist_disc = Categorical(logits=disc_logits)
   disc_action ~ dist_disc.sample()

4. Continuous Head:
   cont_mean = sigmoid(Linear(c_t))                    → (n_cont,) ∈ [0,1]
   dist_cont = Normal(cont_mean, exp(log_std))
   cont_action ~ dist_cont.sample()

5. 输出:
   action   = [disc_logits.detach() ‖ cont_action.detach()]
   log_prob = log_prob_disc + log_prob_cont
   h_next   = (1, 1, 64)
```

#### 隐状态初始化 (`init_hidden`)

GRU 隐状态从全局图编码初始化，而非零初始化：

```
graph_enc (64,) → Linear(64→64) → ReLU → reshape → h_0 (1, 1, 64)
```

#### 批量评估 (`evaluate`)

用于 PPO 更新阶段，输入为 batch 张量，支持给定动作的 log_prob 和 entropy 计算。隐状态形状调整为 `(1, B, 64)`。

### 3.3 Critic — 集中式价值网络

**文件**: `models/critic.py`

采用 CTDE (Centralized Training, Decentralized Execution) 范式，Critic 在训练时可以观测全局状态。

```
输入: global_obs (148,) — M=4 个智能体 obs 拼接 [obs_0 ‖ obs_1 ‖ obs_2 ‖ obs_3]
  → Linear(148 → 128) → ReLU
  → GRU(input=128, hidden=64)
  → Linear(64 → 1)
  → value (1,)
```

Critic 同样使用 GRU 维护时序隐状态 `h_V`。

### 3.4 MAPPOAgentV2 — 智能体封装

**文件**: `models/v2/agent_v2.py`

每个智能体封装了编码器引用、Actor、Critic 及 Episode 级缓存。

#### 核心方法

| 方法 | 调用时机 | 功能 |
|------|---------|------|
| `encode()` | Episode 开始时调用一次 | 运行 GNN 编码器，缓存 `node_embs`, `server_embs`, `graph_enc`，计算拓扑排序，初始化 GRU 隐状态 |
| `act()` | 每步决策时调用 | 获取当前任务节点嵌入、聚合上游决策、调用 Actor 推理动作；返回后 `step_idx` 已自增，可由调用方读取 `topo_order[step_idx - 1]` 得到本次 `task_id` |
| `get_value()` | 每步决策时调用 | 调用 Critic 估计全局状态价值 |
| `evaluate_actions(…, task_ids_batch)` | PPO 更新时调用 | 批量重新计算 log_prob、entropy、value；需传入 `task_ids_batch(B,)` 以按 task_id 索引 node_embs，保证与 `act()` 的 embedding 一致 |
| `reset_episode()` | Episode 结束时调用 | 清空缓存、重置隐状态 |

#### 上游决策聚合 (`L_us_agg`)

在 `act()` 中，智能体收集已决策的拓扑前驱节点的动作，进行平均池化：

```python
upstream_actions = [self.decisions[t] for t in self.topo_order[:self.step_idx] if t in self.decisions]
# Pad/truncate to 64 dims → mean pooling → L_us_agg (64,)
```

这使得每步决策都能感知前驱任务的调度结果，实现 DAG 依赖传播。

#### 拓扑排序

使用 Kahn 算法 (`_kahn_topo_sort`) 对 DAG 进行拓扑排序，决定任务节点的决策顺序。若检测到环则退化为顺序遍历。

---

## 4. 缓冲区设计

**文件**: `utils/buffer.py`

### 4.1 Transition 数据结构

| 字段 | 形状 | 说明 |
|------|------|------|
| `obs` | `(37,)` | 智能体局部观测 |
| `action` | `(8,)` | 执行的动作 |
| `reward` | scalar | 即时奖励 |
| `h_pi` | `(1, 1, 64)` | Actor GRU 隐状态快照 |
| `h_V` | `(1, 1, 64)` | Critic GRU 隐状态快照 |
| `global_obs` | `(148,)` | 全局状态（CTDE） |
| `done` | bool | Episode 结束标志 |
| `log_prob` | scalar | 旧策略的 log 概率 |
| `advantage` | scalar | GAE 优势（在顺序轨迹上预计算后写入） |
| `ret` | scalar | GAE 回报（advantage + V(s)） |
| `task_id` | int | 本次决策对应的 DAG 任务节点索引 |
| `agent_id` | int | 记录该 transition 属于哪个智能体（用于 PPO 更新时按 agent 分组） |

### 4.2 AgentBuffer

每个智能体维护一个 `AgentBuffer`（ξ_i），用于在 Episode 过程中暂存该智能体的 Transition 序列。Episode 结束后，所有 Transition 被倒入 `GlobalBuffer`。

### 4.3 GlobalBuffer

全局经验池 MB，容量 50000，先进先出。提供：

- `add_from_agent_buffer()`: 从 AgentBuffer 倒入数据（含预计算的 `advantage`、`ret`、`task_id`、`agent_id`）
- `sample(batch_size)`: 随机采样 mini-batch，返回包含 `advantages`、`returns`、`task_ids`、`agent_ids` 的字典
- `compute_returns_and_advantages()`: 在顺序轨迹上计算 GAE 优势和回报（Episode 结束时由 `_run_episode` 调用）
- `clear()`: 清空 buffer，每次 PPO 更新后调用以保持同策略纯洁性（Fix C）

**同策略约束（Fix C）**：PPO 是同策略算法，buffer 中的旧数据会因策略偏移导致 importance sampling 失效。每次 PPO 更新后必须调用 `clear()` 清空 buffer，仅保留当前策略采集的数据。

### 4.4 GAE 优势估计

采用 Generalized Advantage Estimation (GAE)：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中：

$$
\delta_t = r_t + \gamma V(s_{t+1}) \cdot (1 - d_t) - V(s_t)
$$

从后向前递推计算，优势最终做标准化：$\hat{A} \leftarrow (\hat{A} - \mu) / (\sigma + \epsilon)$。

**GAE 必须在每个智能体的完整顺序轨迹上计算**，不能在随机采样的 mini-batch 上计算。原因：GAE 公式中 $V(s_{t+1})$ 必须是时序上「下一步」的价值估计，若轨迹被打乱，$V(s_{t+1})$ 指向随机邻居，returns 方差会爆炸（实测可达 50~200），导致 critic_loss 爆炸至 5000~15000。

实现约束：`compute_returns_and_advantages()` 在 `_run_episode` 末尾、`add_from_agent_buffer()` 之前调用，结果写回各 `Transition.advantage` 和 `Transition.ret`。`_ppo_update` 直接从 batch 读取预计算值，不再重新计算 GAE。

---

## 5. 训练流程

### 5.1 整体训练循环

```
for episode in 1..epochs:
    1. _run_episode()       ← 收集经验
    2. if episode % update_every == 0:
           if buffer >= mini_batch_size:
               _ppo_update()    ← PPO 策略更新（K=ppo_epochs 轮）
               buffer.clear()   ← 清空 buffer（同策略约束）
    3. if episode % log_interval == 0:
           log metrics
    4. if episode % save_interval == 0:
           save checkpoint
```

**更新频率（Fix C）**：每 `update_every`（默认 5）个 episode 执行一次 PPO 更新，而非每个 episode 更新。累积多个 episode 的数据可增大有效 batch 规模，提升梯度估计稳定性。

### 5.2 Episode 执行流程 (`_run_episode`)

```
┌─────────────────────────────────────────────────────────┐
│ 0. 重置决策计数: _agent_decision_count[:] = 0            │
│                                                          │
│ 1. 环境重置: env.reset() → obs_dict                     │
│ 2. 智能体重置: agent.reset_episode()                     │
│ 3. 缓冲区清空: agent_buffers[m].clear()                  │
│                                                          │
│ 4. 编码阶段 (Encode-Once):                               │
│    for each agent m:                                     │
│        构建 DAG + 资源图张量                              │
│        agent.encode(dag_x, dag_ei, res_x, res_ei)       │
│        → 缓存 node_embs, server_embs, graph_enc          │
│        → 初始化 h_pi, 计算拓扑排序                        │
│        → 缓存 DAG tensors 到 _dag_tensors[m]（供 PPO 更新使用）│
│                                                          │
│ 5. 异步决策循环:                                         │
│    while not done:                                       │
│        global_obs = concat(all agents' obs)               │
│        available_agents = {m | global_step ≥ local_clock[m]} │
│                                                          │
│        for m in available_agents:                        │
│            action, log_prob, h_pi = agent.act(obs_m)     │
│            current_task_id = topo_order[step_idx - 1]   │  ← 记录任务节点
│            value, h_V = agent.get_value(global_obs)      │
│            计算执行时隙 (geometric distribution)           │
│            更新 local_clock[m]                           │
│            存储 Transition(task_id=current_task_id, agent_id=m) 到 AgentBuffer[m] │
│                                                          │
│        next_obs, rewards, done, info = env.step(actions) │
│        回填 reward 和 done 到对应 Transition              │
│                                                          │
│ 6. Episode 结束，GAE 预计算 (入 GlobalBuffer 之前):       │
│    for each agent m:                                     │
│        用 agents[m].critic 对顺序轨迹估计 values_m        │
│        adv_m, ret_m = compute_returns_and_advantages(    │
│            rewards_m, values_m, dones_m)                 │
│        写回 buf.transitions[i].advantage / .ret          │
│                                                          │
│ 7. 将所有 AgentBuffer 倒入 GlobalBuffer                  │
│    avg_reward = total_reward / episode_decisions         │  ← 除以本 episode 决策数
└─────────────────────────────────────────────────────────┘
```

### 5.3 异步决策机制

AMAPPOv2 的关键创新之一是**异步决策**。每个智能体维护一个 `local_clock`，当全局步数 `global_step ≥ local_clock[m]` 时，该智能体才进行决策。

执行时隙的计算基于任务计算量：

```python
c_cycles = dag.nodes[task_node].get("C", 1.0)
mean_slots = max(1, int(c_cycles / 0.5))
exec_slots = max(1, np.random.geometric(1.0 / mean_slots))
local_clock[m] = global_step + exec_slots
```

这使得计算量大的任务在逻辑上占用更多时间步，模拟真实的异步执行环境。

### 5.4 PPO 更新流程 (`_ppo_update`)

```
┌─────────────────────────────────────────────────────────┐
│ 对 ppo_epochs 轮执行（Fix D: 多 epoch 更新提升样本利用率）│
│                                                          │
│ 1. 从 GlobalBuffer 采样 mini-batch                       │
│    batch = {obs, actions, h_pi, h_V, global_obs,        │
│             log_probs, advantages, returns,              │
│             task_ids, agent_ids}                         │
│                                                          │
│ 2. 直接使用预计算的 GAE 优势和回报:                       │
│    advantages_t = batch["advantages"]   ← 顺序轨迹预计算  │
│    returns_t    = batch["returns"]      ← 顺序轨迹预计算  │
│    advantages_t = normalize(advantages_t)                │
│                                                          │
│ 3. optimizer.zero_grad()  ← 仅一次（Fix A: 合并梯度）    │
│                                                          │
│ 4. 对每个智能体执行 PPO 评估（Fix A: 累加梯度不 step）:  │
│    for agent_id, agent in agents:                        │
│        # Fix B: 使用该智能体自己的 DAG 编码               │
│        dag_x, dag_ei, res_x, res_ei =                   │
│            _get_dag_tensors(agent_id)                    │
│                                                          │
│        # 按 agent_id 过滤，确保 DAG 编码与数据匹配       │
│        mask = (agent_ids == agent_id)                    │
│        if mask.sum() == 0: continue                      │
│                                                          │
│        log_probs, entropies, values, _ =                 │
│            agent.evaluate_actions(obs[mask], actions[mask]│
│                global_obs[mask], h_pi[mask], h_V[mask],  │
│                dag_x, dag_ei, res_x, res_ei,            │
│                task_ids[mask])  ← 按该 agent 的 DAG 编码  │
│                                                          │
│        ratio = exp(log_probs_new - log_probs_old[mask])  │
│        surr1 = ratio × advantages[mask]                  │
│        surr2 = clamp(ratio, 1-ε, 1+ε) × advantages[mask]│
│        actor_loss  = -min(surr1, surr2).mean()           │
│        critic_loss = MSE(values, returns[mask])          │
│        entropy     = entropies.mean()                    │
│                                                          │
│        loss = actor_loss + 0.5×critic_loss - 0.01×entropy│
│        loss.backward()     ← 累加梯度（不在循环内 step） │
│                                                          │
│ 5. 统一裁剪 + 更新（Fix A: 所有 agent 梯度累加后统一 step）│
│    clip_grad_norm_(max_grad_norm)                        │
│    optimizer.step()                                      │
│                                                          │
│ 6. PPO 更新后清空 buffer（Fix C: 同策略约束）            │
│                                                          │
│ 7. 返回平均 loss 指标（除以 agent 数 × ppo_epochs）      │
└─────────────────────────────────────────────────────────┘
```

#### PPO Clip 目标

$$
L^{CLIP} = -\mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t,\;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。

#### 总损失函数

$$
L = L^{CLIP} + 0.5 \cdot L^{VF} - 0.01 \cdot H(\pi)
$$

- $L^{VF} = \text{MSE}(V_\theta(s), R_t)$ — 价值函数损失
- $H(\pi)$ — 策略熵（鼓励探索）

---

## 6. 环境模型

### 6.1 SECEnv — 卫星边缘计算环境

**文件**: `env/sec_env.py`

#### 四层架构

```
IoTD (N=100) → UAV (M=4, agents) → LEO Satellite (K=8) → Cloud (1)
```

#### 观测空间 (37维)

| 区域 | 维度 | 范围 | 说明 |
|------|------|------|------|
| 任务特征 | 0-4 (5维) | — | `[D_in, D_out, C, deadline_rem, topo_pos]` |
| 上游决策 | 5-24 (20维) | — | 5个前驱 × 4维属性 `[D_in, D_out, C, is_done]` |
| 服务器状态 | 25-28 (4维) | — | 4个服务器的 EMA 负载 `[local, UAV, sat, cloud]` |
| 前一动作 | 29-36 (8维) | — | 上一步的动作向量 |

#### 动作空间 (8维)

| 区域 | 维度 | 类型 | 说明 |
|------|------|------|------|
| 卸载 logits | 0-3 (4维) | 离散 | `[local, UAV, satellite, cloud]` 取 argmax |
| 带宽分配 | 4 (1维) | 连续 | sigmoid 压缩到 (0.01, 1.0) |
| 计算分配 | 5 (1维) | 连续 | sigmoid 压缩到 (0.01, 1.0) |
| 位移 | 6-7 (2维) | 连续 | ×50 m/步，裁剪到区域边界 |

#### 奖励函数

$$
r_m = -\eta_t \cdot T_i - \eta_e \cdot \alpha_E \cdot E_i - \lambda_c \cdot \sum_{j=1}^{5} \Phi_j
$$

其中：
- $T_i = T_{trans} + T_{comp}$ — 任务完成时间
- $E_i = E_{tx} + E_{comp}$ — 能耗
- $\alpha_E = 0.01$ — 能量缩放因子（`_E_SCALE`），将 $E_i$ 压缩到与 $T_i$ 同量级
- $\Phi_j$ — 5 类约束违反指示器

**量级说明**：Cloud offloading 场景下，$E_{comp} = \kappa \cdot C \cdot f^2 \approx 10\,\text{J}$，而 $T_{comp} \approx 0.1\,\text{s}$，导致 $\eta_e \cdot E_i$ 比 $\eta_t \cdot T_i$ 大约 100 倍。不加 $\alpha_E$ 时策略梯度被能量项主导，时延优化的梯度信号被淹没。$\alpha_E = 0.01$ 使两项量级之比保持在 0.1~10 之间。

#### 约束违反 (5类)

| 编号 | 约束 | 条件 |
|------|------|------|
| Φ₁ | 任务截止期 | $T_i > T_{max}$ |
| Φ₂ | UAV 碰撞 | 与其他 UAV 距离 < $d_{min}$ |
| Φ₃ | 超出区域 | UAV 移出任务区域 |
| Φ₄ | 速度超限 | 实际速度 > $v_{max}$ |
| Φ₅ | 资源过载 | 计算分配 > 0.95 |

#### 传输时间模型

| 卸载目标 | 传输路径 |
|---------|---------|
| Local | 无传输，$T_{trans} = 0$ |
| UAV | IoTD → UAV (G2U) |
| Satellite | IoTD → UAV → Sat (G2U + U2S) |
| Cloud | IoTD → UAV → Sat → Cloud (G2U + U2S + S2C) |

---

## 7. 参数配置

**文件**: `utils/config.py`

### 7.1 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N` | 100 | IoTD 数量 |
| `M` | 4 | UAV/智能体数量 |
| `K` | 8 | LEO 卫星数量 |
| `J` | 20 | 每 DAG 任务数 |
| `area_size` | 1000.0 | 区域边长 (m) |
| `dt` | 1.0 | 时间步长 (s) |
| `max_steps` | 200 | 每 Episode 最大步数 |
| `H_uav` | 50.0 | UAV 高度 (m) |
| `v_max` | 30.0 | UAV 最大速度 (m/s) |
| `d_min` | 3.0 | 最小安全距离 (m) |

### 7.2 奖励参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `eta_t` | 0.5 | 时间权重 |
| `eta_e` | 0.5 | 能耗权重 |
| `lambda_c` | 10.0 | 约束违反惩罚系数 |

### 7.3 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ |
| `eps_clip` | 0.2 | PPO 裁剪 ε |
| `lr` | 5e-4 | 学习率 |
| `mini_batch_size` | 128 | Mini-batch 大小 |
| `epochs` | 1500 | 训练 Episode 数 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |
| `ppo_epochs` | 4 | PPO 每次更新的 epoch 数（Fix D） |
| `update_every` | 5 | 累积多少 episode 后执行一次 PPO 更新（Fix C） |

### 7.4 网络参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gru_hidden` | 64 | GRU 隐状态维度 |
| `gnn_out_dim` | 128 | GNN 输出维度 |
| `obs_dim` | 37 | 单智能体观测维度 |
| `action_dim` | 8 | 动作维度 |

---

## 8. 数据流图

### 8.1 Episode 数据流

```
SECEnv.reset()
    │
    ▼
_build_graph_inputs_v2(env, m)  ←─ 构建 DAG/资源图张量
    │                              (对每个智能体 m)
    ▼
agent.encode(dag_x, dag_ei, res_x, res_ei)
    │
    ├── GNNEncoderV2.forward()
    │       ├── node_embs   (N, 64)  ── 缓存
    │       ├── server_embs (4, 64)  ── 缓存
    │       └── graph_enc   (64,)    ── 缓存
    │
    ├── _kahn_topo_sort()           ── 缓存 topo_order
    └── ActorV2.init_hidden(graph_enc) ── 初始化 h_pi

    循环每步:
    │
    ├── agent.act(obs_m)
    │       ├── task_id = topo_order[step_idx]   ── 读取后 step_idx 自增
    │       ├── h_v_t = node_embs[task_id]        ── 当前任务节点嵌入
    │       ├── server_agg = mean(server_embs)    ── 服务器聚合
    │       ├── L_us_agg = mean(upstream decisions)── 上游决策聚合
    │       ├── a_prev = obs[29:37]               ── 前一动作
    │       └── ActorV2.forward(h_v_t, L_us_agg, server_agg, a_prev, h_pi, node_embs)
    │               → action, log_prob, h_pi_next
    │
    ├── current_task_id = topo_order[step_idx - 1]  ← 记录本次决策的任务节点
    │
    ├── agent.get_value(global_obs)
    │       └── Critic.forward(global_obs, h_V)
    │               → value, h_V_next
    │
    ├── Transition(obs, action, reward=0, h_pi, h_V, global_obs,
    │             done=False, log_prob, task_id=current_task_id)
    │       → AgentBuffer[m].add(t)
    │
    └── env.step(action_dict) → 回填 reward & done

    Episode 结束，GAE 预计算 (入 GlobalBuffer 前):
    │
    ├── for each agent m:
    │       rewards_m, dones_m ← buf.transitions
    │       values_m ← agents[m].critic(顺序推理，no_grad)
    │       adv_m, ret_m = compute_returns_and_advantages(rewards_m, values_m, dones_m)
    │       写回 buf.transitions[i].advantage / .ret
    │
    └── GlobalBuffer.add_from_agent_buffer(buf)  ← 此后 buf 中含 advantage/ret/task_id
```

### 8.2 PPO 更新数据流

```
GlobalBuffer.sample(mini_batch_size)
    │
    ▼
batch = {obs(B,37), actions(B,8), h_pi(B,1,1,64), h_V(B,1,1,64),
         global_obs(B,148), log_probs(B,),
         advantages(B,),   ← 预计算（顺序轨迹）
         returns(B,),      ← 预计算（顺序轨迹）
         task_ids(B,),     ← 每样本对应的 DAG 节点索引
         agent_ids(B,)}    ← 每样本对应的智能体编号
    │
    ├── advantages_t = normalize(batch["advantages"])
    │
    └── for agent_id, agent in agents:        ← Fix A: 累加梯度
            # Fix B: 使用该智能体的 DAG 编码（从 _dag_tensors 缓存获取）
            dag_x, dag_ei, res_x, res_ei = _get_dag_tensors(agent_id)
            │
            ├── mask = (agent_ids == agent_id)  ← 按 agent_id 过滤
            │
            ├── agent.evaluate_actions(obs[mask], actions[mask],
            │       global_obs[mask], h_pi[mask], h_V[mask],
            │       dag_x, dag_ei, res_x, res_ei,
            │       task_ids[mask])          ← task_id 驱动 node_embs 索引
            │       │
            │       ├── GNNEncoderV2.forward() ── 重新编码（训练模式，使用该 agent 的 DAG）
            │       │
            │       ├── h_v_t_batch = node_embs[task_ids]  ← 按 task_id 精确索引
            │       │
            │       ├── ActorV2.evaluate(...)
            │       │       → log_probs, entropies
            │       │
            │       └── Critic.forward(global_obs, h_V)
            │               → values
            │
            ├── PPO loss 计算 → loss.backward()  ← 累加梯度，不在循环内 step
            │
        clip_grad_norm_()  ← Fix A: 所有 agent 梯度累加后统一裁剪
        optimizer.step()   ← Fix A: 统一更新
        buffer.clear()     ← Fix C: 清空 buffer 保持同策略
```

---

## 9. 关键设计决策

### 9.1 Encode-Once Decode-Many

传统方法在每个决策步都重新运行 GNN 编码器。AMAPPOv2 在 Episode 开始时仅编码一次，后续每步直接使用缓存的 `node_embs` 进行注意力解码。这样做的好处：

1. **计算效率**: 避免 DAG 结构不变时重复编码
2. **一致性**: 同一 Episode 内嵌入表示稳定
3. **注意力聚焦**: 通过 GRU 隐状态动态关注不同节点

**注意**: PPO 更新时仍会重新编码（训练模式），以保证梯度可传播。

### 9.2 节点级注意力 vs 图级编码

AMAPPOv1 使用图级编码（单一向量表示整个 DAG），而 AMAPPOv2 保留节点级嵌入，通过点积注意力动态选择当前关注的节点：

$$
\alpha_i = \frac{\exp(w_t \cdot e_i)}{\sum_j \exp(w_t \cdot e_j)}, \quad c_t = \sum_i \alpha_i \cdot e_i
$$

这使得策略网络能根据 GRU 隐状态 `$w_t$` 的变化，在不同决策步关注不同的任务节点。

### 9.3 异构智能体支持

通过 `agent_type` 参数，`ActorV2` 支持不同动作空间维度：

- **LEO**: 离散 4 维 + 连续 3 维 = 7 维动作
- **UAV**: 离散 4 维 + 连续 5 维 = 9 维动作

共享的 GRU 和注意力机制，仅在输出头区分维度。

### 9.4 参数共享策略

- **共享**: `GNNEncoderV2`（所有智能体共享同一编码器）
- **独立**: 每个 `ActorV2` 和 `Critic` 有独立参数
- **统一优化器**: 所有参数使用单一 Adam 优化器

### 9.5 实现约束（v2.1 修复记录）

以下约束来自 v2.0/v2.1 训练发散的根因分析（调试报告: `docs/superpowers/plans/2026-04-22-amappov2-debug-plan.md` 及 `2026-04-23-amappov2-debug-plan.md`），**必须严格遵守**：

#### 约束 C1：GAE 必须在顺序轨迹上计算

GAE 公式依赖时序连续性（$V(s_{t+1})$ 必须是真实下一步的估计）。将 GAE 放在随机采样的 mini-batch 上计算会导致 returns 方差爆炸，`critic_loss` 膨胀至 5000~15000。

**规则**：`compute_returns_and_advantages` 仅在 `_run_episode` 末尾、对每个 AgentBuffer 的完整顺序轨迹调用。`_ppo_update` 直接读取 `Transition.advantage` 和 `Transition.ret`，不得重新计算 GAE。

#### 约束 C2：`evaluate_actions` 必须按 task_id 索引 node_embs

PPO ratio 计算要求新旧策略在**相同输入特征**下评估同一动作。`act()` 使用 `node_embs[task_id]` 作为 GRU 输入；`evaluate_actions` 如果固定使用 `node_embs[0]`，当 `task_id ≠ 0` 时新旧策略输入空间不同，ratio 在 batch 上均值趋近于 1，PPO 梯度消失，`actor_loss ≈ 0`。

**规则**：每个 Transition 必须记录 `task_id`；`evaluate_actions` 签名必须包含 `task_ids_batch` 参数，并使用 `node_embs[task_ids]` 索引。

#### 约束 C3：avg_reward 必须除以本 episode 的决策数

`_agent_decision_count` 应在每个 `_run_episode` 开始时清零。日志打印时使用当前 episode 末尾的计数值（不累积多个 episode）。

#### 约束 C4：奖励中能量项必须加缩放因子

Cloud offloading 场景下 $E_{comp} \approx 10\,\text{J}$，而 $T_{comp} \approx 0.1\,\text{s}$，导致能量梯度信号约为时延信号的 100 倍。必须使用 `_E_SCALE = 0.01`（或等效调整 `eta_e`）使两项量级相当，否则策略无法同时优化时延和能耗。

#### 约束 C5：PPO 更新必须合并所有 agent 的梯度后统一 step（Fix A）

全部 agent 共享 `shared_encoder` 和同一个 optimizer。若在 agent 循环内对每个 agent 分别 `zero_grad()` → `backward()` → `step()`，则后一个 agent 的 `zero_grad()` 会清除前一个 agent 对 encoder 累积的梯度，最终 encoder 只接收最后一个 agent 的梯度信号，前 75% 梯度信息被丢弃。

**规则**：`zero_grad()` 只在 agent 循环外调用一次；所有 agent 的 `loss.backward()` 累加梯度；循环结束后统一 `clip_grad_norm_()` + `optimizer.step()`。

#### 约束 C6：PPO 更新时每个 agent 必须使用自己的 DAG 编码（Fix B）

`GlobalBuffer` 存放了所有 agent 的 transitions。每个 agent 采集时使用自己的 DAG 生成 `log_prob_old`。若 PPO 更新时统一使用 agent 0 的 DAG 重新编码，则 agent 1~3 的 `node_embs[task_ids]` 基于错误的 DAG，`log_prob_new` 与 `log_prob_old` 的输入特征不匹配，ratio 变为随机噪声，PPO surrogate 梯度方向不一致。

**规则**：Episode 编码阶段必须缓存每个 agent 的 DAG tensors 到 `_dag_tensors[m]`；PPO 更新时通过 `_get_dag_tensors(agent_id)` 获取对应 agent 的 DAG；按 `agent_ids` 做 mask 过滤，确保每个 agent 只用自己 DAG 编码的 node_embs 评估自己的 transitions。

#### 约束 C7：PPO 更新后必须清空 GlobalBuffer（Fix C）

PPO 是同策略（on-policy）算法，importance sampling 修正仅在策略偏移很小时有效（clip 到 1±0.2）。若 buffer 保留多个 episode 前的旧数据，`log_prob_old` 来自历经多次策略更新的旧策略，ratio 可能爆炸或趋于 0，clip 无法修正。

**规则**：每次 PPO 更新后必须调用 `global_buffer.clear()`；训练循环按 `update_every` 累积 N 个 episode 后统一更新并清空。

#### 约束 C8：PPO 每次更新应执行多个 epoch（Fix D）

标准 PPO 在每次数据采集后进行 K 个 epoch 的更新（通常 K=4~10）。若每 episode 仅用 1 个 mini-batch 更新一次，每条 transition 平均参与训练 < 1 次，样本利用率极低，critic 收敛速度慢。

**规则**：`_ppo_update` 外层循环 `ppo_epochs`（默认 4）轮，每轮采样新的 mini-batch 进行更新。

---

## 10. 训练入口

**文件**: `experiments/train_v2.py`

```bash
python experiments/train_v2.py \
    --epochs 1500 \
    --seed 42 \
    --device cuda \
    --lr 5e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --eps_clip 0.2 \
    --mini_batch_size 128 \
    --max_grad_norm 0.5 \
    --ppo_epochs 4 \
    --update_every 5 \
    --log_interval 100 \
    --save_interval 100
```

支持通过命令行参数覆盖 `Config` 中的所有超参数。

---

## 11. 检查点保存

检查点包含以下状态：

```python
state = {
    "episode": episode,
    "shared_encoder": GNNEncoderV2.state_dict(),
    "optimizer": Adam.state_dict(),
    "actor_0": ActorV2.state_dict(),   # 每个智能体
    "critic_0": Critic.state_dict(),    # 每个智能体
    ...
}
```

保存路径: `checkpoints/amappo_v2_ep{episode}.pt`
