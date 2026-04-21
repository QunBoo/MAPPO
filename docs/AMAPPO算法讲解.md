# AMAPPO 算法讲解：架构设计与完整流程

> 本文档基于对项目源码的完整梳理，系统性地讲解 AMAPPO（Asynchronous Multi-Agent Proximal Policy Optimization）的场景建模、网络架构、算法机制和训练流程。

---

## 目录

1. [问题场景与系统模型](#1-问题场景与系统模型)
2. [环境设计](#2-环境设计)
3. [网络架构](#3-网络架构)
4. [算法核心：异步多智能体PPO](#4-算法核心异步多智能体ppo)
5. [基线对比：同步MAPPO](#5-基线对比同步mappo)
6. [训练流程](#6-训练流程)
7. [AMAPPO与MAPPO的核心差异](#7-amappo与mappo的核心差异)
8. [实现细节与局限性说明](#8-实现细节与局限性说明)

---

## 1. 问题场景与系统模型

### 1.1 四层卫星边缘计算（SEC）架构

```
IoT设备（N=100）
      │  G2U链路（Rician衰落）
      ▼
  UAV（M=4，智能体）
      │  U2S链路（Shadowed-Rician衰落）
      ▼
 LEO卫星（K=8）
      │  S2C链路（自由空间路径损耗）
      ▼
    云端（1个）
```

每个 UAV 作为一个 **独立的 MARL 智能体**，管理其覆盖下的 IoT 设备群。UAV 负责：
- **卸载决策**：将 IoT 任务调度至本地/UAV/卫星/云端
- **资源分配**：分配带宽比例和计算资源比例
- **轨迹规划**：控制自身的二维平面位移

### 1.2 任务工作负载：DAG 模型

IoT 设备的计算任务被建模为有向无环图（DAG），捕捉任务间的数据依赖关系。

```
虚拟源点(0)
    │
  [任务A] ──► [任务C]
  [任务B] ──► [任务C] ──► [任务D] ──► 虚拟汇点(J+1)
```

- **节点数**：J=20 个真实任务节点 + 虚拟源/汇
- **分层结构**：3~5 层，每层至少一个节点
- **节点属性**：
  - `D_in` ∈ [0.8, 4.0] MB：输入数据量
  - `D_out` ∈ [0.4, 1.0] MB：输出数据量
  - `C` ∈ [1.0, 3.0] Gcycles：计算量

DAG 的拓扑序（Kahn 算法）决定了任务的执行顺序，前驱任务完成后后继任务才能调度。

### 1.3 信道模型

| 链路类型 | 信道模型 | 带宽 | 载频 |
|---|---|---|---|
| G2U（地面到UAV）| Rician 衰落（K因子与仰角相关）| 20 MHz | 2.4 GHz |
| U2S（UAV到卫星）| Shadowed-Rician 衰落 | 15 MHz | 26 GHz |
| S2C（卫星到云端）| 自由空间路径损耗 | 1 GHz | 26 GHz |

信道容量由香农公式计算：`R = B · log₂(1 + SNR)`

---

## 2. 环境设计

### 2.1 观测空间（37维 per agent）

每个 UAV 智能体的局部观测向量 `obs ∈ ℝ³⁷`，由四部分拼接：

```
obs = [ task_features | upstream_info | server_states | prev_action ]
        [0:5]           [5:25]           [25:29]         [29:37]
```

| 切片 | 维度 | 含义 |
|---|---|---|
| `obs[0:5]` | 5 | 当前待调度任务：D_in, D_out, C, 剩余截止时间, 拓扑位置 |
| `obs[5:25]` | 20 | 最多5个前驱任务的信息（每个4维：D_in, D_out, C, 完成标志）|
| `obs[25:29]` | 4 | 四层服务器（本地/UAV/卫星/云端）的指数移动平均（EMA）计算负载 |
| `obs[29:37]` | 8 | 上一时刻的完整动作向量 |

### 2.2 动作空间（8维 per agent）

每个 UAV 输出混合动作 `action ∈ ℝ⁸`：

```
action = [ offload_logits | bw_frac | cpu_frac | delta_x | delta_y ]
           [0:4]            [4]        [5]         [6]        [7]
```

| 切片 | 维度 | 含义 | 后处理 |
|---|---|---|---|
| `action[0:4]` | 4 | 卸载目标的 logits | argmax → {本地, UAV, 卫星, 云端} |
| `action[4]` | 1 | 带宽分配比例 | sigmoid → [0.01, 1.0] |
| `action[5]` | 1 | 计算资源比例 | sigmoid → [0.01, 1.0] |
| `action[6:8]` | 2 | UAV 位移 Δx, Δy | ×50 m/step，边界裁剪 |

### 2.3 多跳传输时延

任务卸载的传输路径和时延取决于目标层：

| 卸载目标 | 传输路径 | 时延计算 |
|---|---|---|
| 本地 | 无传输 | T_trans = 0 |
| UAV | G2U（单跳）| T_trans = D_in / R_G2U |
| 卫星 | G2U + U2S（两跳）| T_trans = D_in/R_G2U + D_out/R_U2S |
| 云端 | G2U + U2S + S2C（三跳）| T_trans = D_in/R_G2U + .../R_U2S + .../R_S2C |

### 2.4 奖励函数

```
r_m = -η_t · T_i  -  η_e · E_i  -  λ · Σ Φ_k
```

- **时延代价** `T_i = T_trans + T_comp`（传输 + 计算时延）
- **能耗代价** `E_i = E_tx + E_comp`（发射能耗 + 计算能耗）
  - 计算能耗采用立方模型：`E_comp = κ · C · f²`
- **约束惩罚** `λ = 10.0`，触发条件 Φ_k 包括：
  1. 超过任务截止期限
  2. UAV 间碰撞（距离 < 3 m）
  3. 飞出覆盖区域
  4. 超过最大飞行速度
  5. 服务器计算负载超过 95%

权重：`η_t = η_e = 0.5`（时延与能耗等权重）

### 2.5 Episode 流程

```
reset() → 生成 N 个 IoT 的 DAG，初始化 UAV 位置
  │
  ▼ 循环（max_steps=200）
step(action_dict)
  │
  ├── 每个 Agent 按 DAG 拓扑序处理当前任务
  ├── 执行卸载与资源分配
  ├── 计算时延、能耗、奖励
  ├── 更新 UAV 位置
  └── 检查终止（所有 DAG 完成 or 达到最大步数）
```

---

## 3. 网络架构

### 3.1 整体架构概览

```
观测 obs(37维)
    │
    ├── task_features(5维) ──┐
    │                        │
    ├── DAG 图结构 ──────────► GNNEncoder(128维) ──┐
    │                        │                      │
    └── resource_graph ───────┘                      │
                                                     ▼
    upstream_info(20维) ──────────────────────► 拼接(160维)
    prev_action(8维)                                 │
    server_states(4维)                               ▼
                                               GRU(64维)
                                               /        \
                                 discrete_head           continuous_head
                                  (4维 logits)          (4维: bw,cpu,dx,dy)
                                      │
                               Categorical分布
                                      │
                                  动作采样
```

### 3.2 双路径 GNN 编码器（核心创新）

`GNNEncoder` 是 AMAPPO 在感知层面的核心创新，对**任务DAG**和**资源图**分别进行图神经网络编码。

#### 路径1：任务DAG编码器（TaskDAGEncoder）

```
输入：DAG节点特征矩阵 X ∈ ℝ^{N×5}（每个节点5维：D_in, D_out, C, deadline, position）
      DAG边索引 edge_index（有向边）

双向GraphSAGE设计：
┌────────────────────────────────────────────┐
│  正向边 ──► SAGEConv_fwd_1(5→32)           │
│                                  ──► sum ──► Layer1(32维)
│  反向边 ──► SAGEConv_bwd_1(5→32)           │
└────────────────────────────────────────────┘
                    │  BatchNorm + ReLU
                    ▼
┌────────────────────────────────────────────┐
│  正向边 ──► SAGEConv_fwd_2(32→64)          │
│                                  ──► sum ──► Layer2(64维)
│  反向边 ──► SAGEConv_bwd_2(32→64)          │
└────────────────────────────────────────────┘
                    │  BatchNorm + ReLU
                    ▼
             Max-Pooling（跨所有节点）
                    │
             DAG嵌入 ∈ ℝ^64
```

**双向编码的意义**：
- **正向传播**（沿原始边）：节点汇聚来自前驱的信息，捕捉"我的前驱完成了什么"
- **反向传播**（沿反转边）：节点汇聚来自后继的信息，捕捉"我的后继需要什么"
- 两路求和后，每个节点同时感知上游依赖和下游需求

#### 路径2：资源图编码器（ResourceGraphEncoder）

```
输入：资源节点特征 R ∈ ℝ^{4×2}
      4个节点：[本地, UAV, 卫星, 云端]
      每个节点2维：[当前EMA负载, 归一化CPU容量]

全连接图（4节点，12条有向边）：
Layer1: SAGEConv(2→32) → BatchNorm + ReLU
Layer2: SAGEConv(32→64) → BatchNorm + ReLU
Max-Pooling → 资源嵌入 ∈ ℝ^64
```

#### 联合投影

```
cat([dag_emb, res_emb])  →  ℝ^128
       Linear(128→128) + ReLU
              │
       联合嵌入 ∈ ℝ^128
```

### 3.3 Actor 网络

```python
输入 = cat([gnn_embed(128), upstream_dec(20), prev_action(8), server_embed(4)])
     = 160维

GRU(input=160, hidden=64, batch_first=True)
         │
    ┌────┴────┐
discrete_head  continuous_head
Linear(64→4)   Linear(64→4)
     │              │
Categorical     sigmoid[:2]  # bw, cpu ∈ (0,1)
  采样            raw[2:]   # dx, dy (UAV位移)
```

GRU 的隐藏状态 `h_pi` 在 episode 内的各步之间保持，捕捉时序依赖。

### 3.4 集中式 Critic 网络（CTDE 设计）

```
全局状态 = cat([obs_1, obs_2, ..., obs_M])  →  37×4 = 148维

Linear(148→128) + ReLU
GRU(128→64)
Linear(64→1)
     │
  状态价值 V(s) ∈ ℝ
```

Critic 接受所有智能体的联合观测（全局状态），实现**集中训练、分散执行（CTDE）**。

---

## 4. 算法核心：异步多智能体PPO

### 4.1 异步机制：双时钟设计

AMAPPO 的核心创新是**双时钟异步决策机制**，模拟真实任务执行的时间异步性。

#### 时钟定义

```
全局时钟 t'：主循环每次迭代 +1
局部时钟 t_i：第 i 个智能体下次可以决策的最早全局步
```

#### 异步决策逻辑

```python
while not done:
    available_agents = [m for m if global_step >= local_clocks[m]]

    for m in all_agents:
        if m in available_agents:
            # 智能体 m 执行决策
            action[m], log_prob[m], h_pi[m] = agent[m].act(obs[m], ...)
            value[m] = agent[m].get_value(global_obs)
            buffer[m].add(transition)     # 写入经验缓冲区

            # 根据任务计算量确定下次可用时刻
            C_cycles = current_task[m]['C']           # 计算量（Gcycles）
            mean_slots = max(1, int(C_cycles / 0.5))  # 基础时隙数
            exec_slots = geometric(1.0 / mean_slots)   # 几何分布采样
            local_clocks[m] = global_step + exec_slots # 调度下次决策

        else:
            # 智能体 m 不可用：复用上次动作
            action[m] = last_action[m]   # 不写缓冲区

    env.step(action)  # 环境统一推进
    global_step += 1
```

#### 异步性的物理意义

| 计算量 C (Gcycles) | 平均时隙 mean_slots | 决策频率 | 物理含义 |
|---|---|---|---|
| 0.5 | 1 | 每步决策 | 轻量任务，快速完成 |
| 1.5 | 3 | 约每3步决策 | 中等任务 |
| 3.0 | 6 | 约每6步决策 | 重度计算任务 |

这样，**计算量大的任务使智能体决策频率降低**，符合真实系统中任务执行占用资源的物理规律。

#### 异步决策示意图

```
时间步:    1    2    3    4    5    6    7    8
Agent 0:  [决策]      [决策]           [决策]
Agent 1:  [决策] [决策]      [决策] [决策]
Agent 2:  [决策]           [决策]           [决策]
Agent 3:  [决策] [决策] [决策]      [决策]

↑ 不同智能体根据各自任务计算量，以不同频率异步决策
```

### 4.2 PPO 更新步骤

经验收集结束后，进行 PPO 参数更新：

#### 步骤1：经验采样

```python
batch = global_buffer.sample(batch_size=128)
# batch 包含：obs, action, reward, log_prob_old, value_old, done, global_obs
```

#### 步骤2：计算广义优势估计（GAE）

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
A_t = δ_t + γ · λ_GAE · (1 - done_t) · A_{t+1}
R_t = A_t + V(s_t)
```

参数：γ = 0.99，λ_GAE = 0.95，从后往前递推。

#### 步骤3：优势归一化

```python
A_normalized = (A - A.mean()) / (A.std() + 1e-8)
```

#### 步骤4：计算 PPO 目标

```python
ratio = exp(log_π_new(a|s) - log_π_old(a|s))

# 裁剪目标
L_actor = -min(
    ratio * A,
    clip(ratio, 1-ε, 1+ε) * A
).mean()

# Critic 损失（均方误差）
L_critic = MSE(V_new(s), R)

# 总损失（熵正则化防止策略过早收敛）
L_total = L_actor + 0.5 · L_critic - 0.01 · H(π)
```

裁剪系数 ε = 0.2，限制每次更新的策略变化幅度。

#### 步骤5：梯度更新

```python
optimizer.zero_grad()
L_total.backward()
clip_grad_norm_(parameters, max_norm=0.5)  # 梯度裁剪
optimizer.step()
```

---

## 5. 基线对比：同步MAPPO

### 5.1 同步决策逻辑

```python
while not done:
    for m in all_agents:
        # 每个时间步，所有智能体同时决策
        action[m], log_prob[m], h_pi[m] = agent[m].act(obs[m], ...)
        value[m] = agent[m].get_value(global_obs)
        buffer[m].add(transition)

    env.step(action)
```

### 5.2 与 AMAPPO 的关键区别

| 维度 | 同步MAPPO | 异步AMAPPO |
|---|---|---|
| 决策频率 | 每个智能体每步都决策 | 仅在任务完成后决策 |
| 计算量建模 | 忽略任务执行时间 | 任务计算量决定决策间隔 |
| 经验写入 | 每步都写 | 仅在决策时写 |
| 动作更新 | 每步更新 | 非决策步复用上次动作 |
| 物理真实性 | 较低 | 较高（符合实际执行延迟）|

---

## 6. 训练流程

### 6.1 初始化

```python
config = Config()        # 超参数配置
env = SECEnvironment()   # 四层SEC仿真环境
shared_gnn = GNNEncoder()  # 所有智能体共享的GNN编码器

agents = [MAPPOAgent(shared_gnn, actor_i, critic_i) for i in range(M)]
optimizer = Adam(all_parameters, lr=5e-4)

agent_buffers = [AgentBuffer() for _ in range(M)]
global_buffer = GlobalBuffer(capacity=50000)
```

### 6.2 训练主循环

```
For epoch = 1 to 1500:
    ┌─────────────────────────────────────────┐
    │           数据收集阶段                    │
    │  obs = env.reset()                       │
    │  local_clocks = [0, 0, 0, 0]            │
    │                                          │
    │  While not done:                         │
    │    1. 确定可用智能体                      │
    │    2. 可用智能体执行 act() 和 get_value() │
    │    3. 不可用智能体复用上次动作             │
    │    4. env.step(actions)                  │
    │    5. 可用智能体写入 agent_buffer         │
    │    6. 更新局部时钟                        │
    └─────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────┐
    │           参数更新阶段                    │
    │  global_buffer.add_from_agents(...)      │
    │  batch = global_buffer.sample(128)       │
    │  GAE 计算                                │
    │  PPO 损失计算                            │
    │  梯度更新                                │
    └─────────────────────────────────────────┘
              │
              ▼
    每100轮：记录 TensorBoard 日志
    每100轮：保存检查点
```

### 6.3 关键超参数

| 参数 | 值 | 说明 |
|---|---|---|
| 训练轮数 | 1500 | 总 epoch 数 |
| 学习率 | 5e-4 | Adam 优化器 |
| 折扣因子 γ | 0.99 | 长期回报权重 |
| GAE 参数 λ | 0.95 | 优势估计平滑 |
| PPO 裁剪 ε | 0.2 | 策略更新幅度限制 |
| 批量大小 | 128 | 每次 PPO 更新的样本数 |
| 最大梯度范数 | 0.5 | 梯度裁剪阈值 |
| 熵系数 | 0.01 | 探索鼓励强度 |
| Critic 系数 | 0.5 | Critic 损失权重 |
| 缓冲区容量 | 50000 | 全局经验池大小 |

---

## 7. AMAPPO 与 MAPPO 的核心差异

### 7.1 异步决策是主要创新

```
同步MAPPO:
t=1: [A0决策] [A1决策] [A2决策] [A3决策] → env.step
t=2: [A0决策] [A1决策] [A2决策] [A3决策] → env.step
...（每步全部决策，不考虑任务执行时间）

异步AMAPPO:
t=1: [A0决策] [A1决策] [A2决策] [A3决策] → env.step
t=2: [A0等待]  [A1决策]  [A2等待]  [A3决策] → env.step
t=3: [A0决策] [A1等待] [A2决策]  [A3等待] → env.step
...（根据计算量异步决策，更贴近真实系统）
```

### 7.2 GNN 图结构感知

相比标准 MAPPO 使用平坦 MLP 处理观测，AMAPPO 的 GNN 编码器能够：
- 捕捉 DAG 中任务间的依赖关系（双向GraphSAGE）
- 感知计算资源的全局分布（全连接资源图）
- 通过 Max-Pooling 产生全图级别的表示

### 7.3 物理系统建模的精细化

| 方面 | MAPPO | AMAPPO |
|---|---|---|
| 信道模型 | 简化 | Rician/Shadowed-Rician/FSPL |
| 传输时延 | 单跳 | 多跳（最多G2U+U2S+S2C三跳）|
| 能耗模型 | 线性 | 立方模型（κ·C·f²）|
| 任务结构 | 独立任务 | DAG依赖任务 |
| 决策时机 | 固定频率 | 任务完成驱动 |

---

## 8. 实现细节与局限性说明

### 8.1 当前实现的简化

**简化1：PPO 重要性采样比率**

当前实现中，`log_probs_old` 直接取 `log_probs_new.detach()`，导致比率 ratio ≈ 1，PPO 裁剪实际未生效，退化为普通策略梯度：

```python
# 当前实现（简化）
log_probs_new, entropies, values_new = agent.evaluate_actions(batch)
log_probs_old = log_probs_new.detach()  # ratio = exp(0) = 1

# 正确实现（应从 buffer 中读取采集时的旧 log_prob）
log_probs_old = batch['log_probs']  # 采集时存储的旧策略概率
```

**简化2：单智能体代表更新**

当前训练器只更新 agent 0 的 actor/critic 参数（共享 GNN 会被更新），而非真正的每智能体独立优化：

```python
# 当前实现
loss.backward()   # 只涉及 agent[0] 的参数

# 完整实现应为：每个 agent 维护独立的 optimizer，分别更新
```

**简化3：GNN 评估的广播优化**

在 PPO 更新的 `evaluate_actions` 中，GNN 在代表性图上运行一次，其 128 维嵌入被广播到整个 mini-batch，避免了逐步图构建的开销，但牺牲了不同时间步图结构变化的精确性。

### 8.2 GAE 计算的注意点

`GlobalBuffer.sample()` 进行随机采样，被采样的 transitions 可能不是时序连续的。在这种情况下对随机混合的 batch 做 GAE 计算，严格意义上的时序回溯是近似的。生产实现通常应在完整轨迹上做 GAE，再进行 mini-batch 随机采样。

### 8.3 模型文件对应关系

```
models/gnn_encoder.py  → GNNEncoder, TaskDAGEncoder, ResourceGraphEncoder
models/actor.py        → Actor（GRU + 混合动作头）
models/critic.py       → Critic（集中式价值网络）
models/agent.py        → MAPPOAgent（封装 act/get_value/evaluate_actions）
algorithms/amappo.py   → AMAPPOTrainer（异步经验收集 + PPO更新）
algorithms/mappo.py    → MAPPOTrainer（同步经验收集 + PPO更新，基线）
env/sec_env.py         → SECEnvironment（四层SEC仿真）
env/dag_generator.py   → DAGGenerator（随机分层DAG生成）
env/channel_model.py   → ChannelModel（物理层信道建模）
utils/buffer.py        → Transition, AgentBuffer, GlobalBuffer
utils/config.py        → Config（超参数数据类）
utils/logger.py        → TensorBoardLogger
experiments/train.py   → 训练入口（CLI）
```

---

## 总结

AMAPPO 的核心贡献在于两个层面：

1. **感知层**：双路径 GNN 编码器将任务 DAG 的拓扑依赖关系和计算资源的全局状态同时编码为结构化的图嵌入，使策略网络能够感知任务间的数据流关系。

2. **决策层**：双时钟异步机制让每个智能体的决策频率与其当前任务的计算量挂钩，复现了真实系统中任务执行占用计算资源导致决策延迟的物理现象，使多智能体之间的时序关系更加真实。

这两点结合标准的 MAPPO（CTDE 框架 + GAE + PPO 裁剪）形成了 AMAPPO 的完整方案，面向卫星边缘计算场景下的任务卸载和资源管理联合优化问题。
