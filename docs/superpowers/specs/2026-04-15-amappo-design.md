# AMAPPO算法复现项目设计文档

**日期：** 2026-04-15
**目标：** 核心算法验证——实现AMAPPO，验证其收敛性并与MAPPO（同步版）对比，验证异步机制的有效性
**论文：** Cost-aware Dependent Task Offloading and Resource Allocation for Satellite Edge Computing: An Asynchronous Deep Reinforcement Learning Approach (IEEE TMC, DOI 10.1109/TMC.2025.3645456)

---

## 1. 项目范围与约束

### 实现范围
- **实现：** AMAPPO核心算法 + MAPPO同步对照 + SEC仿真环境 + GNN编码器
- **简化：** 设备关联用随机分配（替代一对一匹配理论），任务排序用标准拓扑排序（替代完整MATS）
- **不实现：** MADDPG、A-PPO、IPPO等其他基线；Alibaba数据集加载；实际卫星物理层

### 数据来源
合成数据：按论文参数范围随机生成DAG（任务数J=20，数据量[0.8,4.0]MB，CPU[1,3]Gcycles），无需外部数据集。

### 成功标准
- AMAPPO训练曲线可见收敛趋势（奖励上升并趋于稳定）
- AMAPPO系统成本低于MAPPO，验证异步机制优势
- 约束违反率随训练降低

---

## 2. 项目结构

```
amappo/
├── env/
│   ├── sec_env.py        # SECEnv：主仿真环境（Gym接口）
│   ├── dag_generator.py  # 合成DAG随机生成
│   └── channel_model.py  # 通信信道模型（Rician、自由空间）
├── models/
│   ├── gnn_encoder.py    # GraphSAGE任务图/网络资源图编码器
│   ├── actor.py          # Actor网络（GRU + 注意力解码器）
│   ├── critic.py         # Critic网络
│   └── agent.py          # Agent封装（actor + critic + 共享GNN）
├── algorithms/
│   ├── amappo.py         # AMAPPO训练器（异步双时钟）
│   └── mappo.py          # MAPPO训练器（同步，用于对比）
├── utils/
│   ├── buffer.py         # 经验缓冲区（per-agent + 全局MB）
│   ├── config.py         # 超参数配置（dataclass）
│   └── logger.py         # TensorBoard日志封装
├── experiments/
│   ├── train.py          # 训练入口（命令行参数）
│   └── plot_results.py   # 收敛曲线与对比图绘制
└── requirements.txt
```

**模块边界原则：**
- `env/`：仅物理仿真，无RL逻辑
- `models/`：仅PyTorch网络定义，无训练循环
- `algorithms/`：仅训练流程，调用env和models
- `experiments/`：入口点，组装所有模块

---

## 3. 仿真环境（SECEnv）

### 系统参数（默认值）

| 参数 | 值 |
|---|---|
| IoTD数量 N | 100 |
| UAV数量 M | 4 |
| LEO卫星数量 K | 8 |
| 云服务器 | 1 |
| 每DAG任务数 J | 20（默认），范围[10,50] |
| 飞行区域 | 1km × 1km |
| UAV飞行高度 | [40, 60] m |

### 简化决策
- **设备关联：** 随机分配（每个IoTD随机关联一个UAV）
- **任务排序：** 标准拓扑排序（Kahn算法）替代完整MATS

### Gym接口

```python
env = SECEnv(config)
obs_dict, info = env.reset()          # 生成新DAG，初始化网络
obs_dict, rew_dict, done, info = env.step(action_dict)  # 一个全局时间槽
```

`obs_dict` 和 `action_dict` 均以 `{agent_id: tensor}` 形式组织。

### 观测空间（per-agent）

每个智能体观测包含以下拼接向量：
- 当前任务原始特征：`[D_in, D_out, C, |Pre|, |Suc|]`（5维）
- 上游任务卸载决策：`(|Pre| × 4)` 维one-hot（零填充到固定长度）
- 可用服务器状态：`[f_avail, current_load]` × 服务器数（每UAV可见1本地+1UAV+K卫星+1云）
- 上一步动作向量

### 动作空间（per-agent，全连续输出，环境内处理离散化）

| 分量 | 类型 | 说明 |
|---|---|---|
| 卸载决策 | softmax(4) → argmax | 本地/UAV/卫星/云 |
| 带宽分配比 | sigmoid → [0,1] | 上行带宽比例 |
| 计算资源分配 | sigmoid → [0,1] | 映射到实际频率范围 |
| UAV位置增量 | tanh → [-1,1]² | 仅UAV智能体；乘最大速度 |

### 奖励函数

```
r_i = -η_t * T_i - η_e * E_i - Σ λ_ι * Φ_ι
```

惩罚项：
- Φ1：延迟约束违反 max(0, T_comp - T_max)
- Φ2：UAV计算资源过载
- Φ3：卫星计算资源过载
- Φ4：UAV速度约束违反
- Φ5：UAV碰撞约束违反

权重：η_t=0.5, η_e=0.5, λ_ι=10（初始值，可调）

### DAG生成器（dag_generator.py）

按论文参数范围生成随机有向无环图：
- 随机层次结构（3-5层），层间随机连边
- 节点属性：D_in ∈ [0.8,4.0]MB，D_out ∈ [0.4,1.0]MB，C ∈ [1,3]Gcycles
- 添加虚拟入口/出口节点（零属性）

### 信道模型（channel_model.py）

| 链路 | 模型 |
|---|---|
| G2U / U2G | Rician衰落（含LoS/NLoS，K因子随仰角变化） |
| G2S / S2G / U2S / S2U | Shadowed-Rician（3种阴影等级：Light/Average/Heavy） |
| ISL / S2C | 自由空间路径损耗 |

---

## 4. 神经网络架构

### GNN编码器（gnn_encoder.py）

**任务DAG编码器：**
- 基于PyTorch Geometric（PyG）实现GraphSAGE变体
- 分离上游邻居聚合和下游邻居聚合（两路独立SAGEConv）
- 2层，ReLU激活，节点嵌入维度64
- 输入特征：`[D_in, D_out, C, |Pre|, |Suc|]`（5维）

**网络资源图编码器：**
- 无向图GNN（SAGEConv），2层
- 节点嵌入维度64
- 输入特征：`[f_avail, current_load]`（2维）

**输出：**
- Max-pooling后接全连接层，输出联合嵌入向量（128维）

### Actor网络（actor.py）

```
输入：联合嵌入(128) + 上游决策 + 上一动作 + 服务器嵌入
  ↓
GRU（input=N_in, hidden=64）
  ↓
注意力机制（对历史任务嵌入序列打分，生成上下文向量）
  ↓
上下文向量(64) → 全连接层
  ↙              ↘
离散头              连续头
Linear→softmax(4)   Linear→sigmoid×N_cont
（卸载决策）        （带宽/算力/UAV位置）
```

### Critic网络（critic.py）

- 输入：全局状态（所有智能体观测的拼接）
- 结构与Actor共享GRU主干设计，但独立参数
- 输出：单标量 V(o)
- 梯度仅通过当前决策智能体自身观测传播

### Agent封装（agent.py）

- 持有共享GNN编码器（actor和critic复用）
- 各智能体维护独立GRU隐状态：`h_π`（actor用）和 `h_V`（critic用）
- 接口：
  - `act(obs) → action, log_prob, h_π`
  - `evaluate(obs, action) → V, log_prob, entropy`

---

## 5. 训练机制

### 5.1 AMAPPO（异步，algorithms/amappo.py）

**双时钟机制：**
- 全局时钟 `t' = 1..GT`：固定步长，主循环控制
- 智能体本地时钟 `t_i`：可变，上一任务完成即触发决策

**异步决策流程（每个全局时间槽）：**
1. 检查每个智能体是否"可用"（上一任务执行时间已到）
2. 可用智能体：用当前观测调用 `agent.act(obs)` 获取动作，立即执行
3. 不可用智能体：跳过，保持上一动作，不写入缓存

**经验缓冲区：**
- 每个智能体维护独立转移缓存 `ξ_i = [(s,o,a,h_π,h_V,r), ...]`
- 每个epoch结束后，所有 `ξ_i` 汇入全局经验池 `MB`

### 5.2 MAPPO（同步，algorithms/mappo.py）

- 每个全局时间槽，**所有**智能体同步决策
- 统一经验缓存，无per-agent独立缓存
- 其余与AMAPPO完全相同（共用PPO更新逻辑）

### 5.3 PPO更新（两者共用）

**超参数：**

| 参数 | 值 |
|---|---|
| 折扣因子 γ | 0.99 |
| GAE平滑参数 φ | 0.95 |
| PPO裁剪参数 ε | 0.2 |
| 学习率 α | 5×10⁻⁴ |
| Mini-batch大小 | 128 |
| GRU隐藏层维度 | 64 |
| 训练轮数 ep | 1500 |
| 梯度裁剪 max_norm | 0.5 |

**更新步骤（每epoch结束）：**
1. 从MB采样mini-batch（size=128）
2. 计算GAE优势估计（式57）
3. 最小化Critic TD误差（式55）
4. 最大化Actor PPO-clip目标（式56）
5. 梯度裁剪后更新参数

---

## 6. 实验与评估

### 训练入口

```bash
python experiments/train.py --algo amappo --epochs 1500 --seed 42
python experiments/train.py --algo mappo  --epochs 1500 --seed 42
```

### 日志与监控（TensorBoard）

每100步记录：
- 回合累积奖励（Episode Reward）
- 系统总延迟 T_total
- 系统总能耗 E_total
- 综合成本 Cost = η_t*T + η_e*E
- 约束违反率（各Φ_ι的违反频率）
- Critic损失、Actor损失、策略熵

每100轮保存checkpoint到 `checkpoints/`

### 绘图（plot_results.py）

- 收敛曲线：AMAPPO vs MAPPO的奖励/成本曲线（均值 ± 标准差，5个随机种子）
- 成本对比柱状图：收敛后最后100轮的平均值

---

## 7. 依赖环境

```
# requirements.txt
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
tensorboard>=2.8.0
gymnasium>=0.26.0
networkx>=2.6.0   # DAG生成与拓扑排序
```

---

## 8. 开发顺序建议

1. `env/dag_generator.py` → 验证DAG生成正确性（可视化图结构）
2. `env/channel_model.py` → 单元测试各信道速率公式
3. `env/sec_env.py` → 验证reset/step接口，检查观测/奖励维度
4. `models/gnn_encoder.py` → 验证编码器输出维度和梯度流
5. `models/actor.py` + `models/critic.py` → 验证前向传播
6. `utils/buffer.py` → 验证数据存取正确性
7. `algorithms/mappo.py` → 先实现同步版，验证PPO更新逻辑
8. `algorithms/amappo.py` → 在MAPPO基础上增加异步机制
9. `experiments/train.py` → 端到端训练，观察收敛
10. `experiments/plot_results.py` → 生成对比图
