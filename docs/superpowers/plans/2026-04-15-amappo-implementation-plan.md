# AMAPPO 实现计划

**日期：** 2026-04-15  
**基于：** [设计文档](../specs/2026-04-15-amappo-design.md)  
**目标：** 按设计文档开发顺序，逐步实现并验证各模块
**运行环境：** conda activate appo
**开发进度：** 已完成Step 1到Step 8，从Step9继续开发

---

## 阶段一：环境模块（env/）

### （已完成）Step 1 — dag_generator.py
**任务：** 实现随机DAG生成器

- 随机层次结构（3-5层），层间随机连边，确保无环
- 节点属性：`D_in ∈ [0.8,4.0]MB`，`D_out ∈ [0.4,1.0]MB`，`C ∈ [1,3]Gcycles`
- 添加虚拟入口/出口节点（零属性）
- 使用 `networkx` 构建图，支持拓扑排序（Kahn算法）

**验收：** 可视化生成的DAG图结构，确认无环、层次合理

---

### （已完成）Step 2 — channel_model.py
**任务：** 实现各链路信道模型

| 链路 | 模型 |
|---|---|
| G2U / U2G | Rician衰落（K因子随仰角变化） |
| G2S / S2G / U2S / S2U | Shadowed-Rician（Light/Average/Heavy） |
| ISL / S2C | 自由空间路径损耗 |

**验收：** 单元测试各信道速率公式，输出合理的传输速率值

---

### （已完成）Step 3 — sec_env.py
**任务：** 实现主仿真环境（Gym接口）

系统参数：N=100 IoTD，M=4 UAV，K=8 LEO，J=20 任务/DAG

- `reset()` → 生成新DAG，随机分配设备关联，返回 `obs_dict`
- `step(action_dict)` → 执行一个全局时间槽，返回 `(obs_dict, rew_dict, done, info)`
- 观测空间：任务特征(5) + 上游决策(|Pre|×4) + 服务器状态 + 上一动作
- 动作空间：卸载决策(softmax 4) + 带宽分配(sigmoid) + 算力分配(sigmoid) + UAV位移(tanh 2)
- 奖励：`r_i = -η_t*T_i - η_e*E_i - Σλ_ι*Φ_ι`，η_t=η_e=0.5，λ_ι=10

**验收：** 验证 reset/step 接口，检查观测/奖励维度正确

---

## 阶段二：神经网络模块（models/）

### （已完成）Step 4 — gnn_encoder.py
**任务：** 实现双路GNN编码器（基于PyTorch Geometric）

- **任务DAG编码器：** GraphSAGE变体，分离上/下游邻居聚合，2层，节点嵌入64维，输入5维
- **网络资源图编码器：** 无向SAGEConv，2层，节点嵌入64维，输入2维
- 输出：max-pooling → 全连接 → 联合嵌入128维

**验收：** 验证输出维度(128)和梯度流正常

---

### Step 5 — actor.py + critic.py
**任务：** 实现Actor和Critic网络

**Actor：**
```
联合嵌入(128) + 上游决策 + 上一动作 + 服务器嵌入
  → GRU(hidden=64)
  → 注意力机制（历史任务嵌入序列）
  → 离散头: Linear→softmax(4)（卸载决策）
  → 连续头: Linear→sigmoid（带宽/算力/UAV位置）
```

**Critic：**
- 输入：全局状态（所有智能体观测拼接）
- 共享GRU主干设计，独立参数
- 输出：单标量 V(o)

**验收：** 验证前向传播，输出维度正确，无梯度异常

---

### Step 6 — agent.py
**任务：** 实现Agent封装

- 持有共享GNN编码器（actor/critic复用）
- 各智能体维护独立GRU隐状态：`h_π`（actor）和 `h_V`（critic）
- 接口：
  - `act(obs) → action, log_prob, h_π`
  - `evaluate(obs, action) → V, log_prob, entropy`

---

## 阶段三：工具模块（utils/）

### Step 7 — buffer.py
**任务：** 实现经验缓冲区

- per-agent 独立转移缓存 `ξ_i = [(s,o,a,h_π,h_V,r), ...]`
- 全局经验池 `MB`，支持从各 `ξ_i` 汇入
- 支持 mini-batch 采样（size=128）

**验收：** 验证数据存取正确性，采样维度一致

---

### Step 8 — config.py + logger.py
**任务：** 实现配置和日志工具

**config.py：** dataclass 封装所有超参数

| 参数 | 值 |
|---|---|
| γ | 0.99 |
| φ (GAE) | 0.95 |
| ε (PPO clip) | 0.2 |
| α (lr) | 5×10⁻⁴ |
| mini-batch | 128 |
| GRU hidden | 64 |
| epochs | 1500 |
| max_norm | 0.5 |

**logger.py：** TensorBoard封装，每100步记录：
- Episode Reward、T_total、E_total、Cost
- 约束违反率（Φ1~Φ5）
- Critic损失、Actor损失、策略熵

---

## 阶段四：训练算法（algorithms/）

### Step 9 — mappo.py
**任务：** 实现同步MAPPO训练器

- 每时间槽所有智能体同步决策
- 统一经验缓存
- PPO更新步骤：
  1. 从MB采样mini-batch
  2. 计算GAE优势估计
  3. 最小化Critic TD误差
  4. 最大化Actor PPO-clip目标
  5. 梯度裁剪（max_norm=0.5）后更新

**验收：** 训练曲线可见学习信号（奖励有上升趋势）

---

### Step 10 — amappo.py
**任务：** 在MAPPO基础上实现异步AMAPPO

**双时钟机制：**
- 全局时钟 `t' = 1..GT`：主循环控制
- 智能体本地时钟 `t_i`：上一任务完成即触发决策

**异步决策流程（每全局时间槽）：**
1. 检查每个智能体是否"可用"（上一任务执行时间已到）
2. 可用智能体：调用 `agent.act(obs)`，立即执行，写入 `ξ_i`
3. 不可用智能体：跳过，不写入缓存

**验收：** 异步机制正常触发，各智能体决策频率不同

---

## 阶段五：实验入口（experiments/）

### Step 11 — train.py
**任务：** 实现训练入口

```bash
python experiments/train.py --algo amappo --epochs 1500 --seed 42
python experiments/train.py --algo mappo  --epochs 1500 --seed 42
```

- 命令行参数解析（algo, epochs, seed, config覆盖）
- 每100轮保存checkpoint到 `checkpoints/`
- 端到端训练，观察收敛

---

### Step 12 — plot_results.py
**任务：** 实现结果可视化

- 收敛曲线：AMAPPO vs MAPPO 奖励/成本曲线（均值 ± 标准差，5个随机种子）
- 成本对比柱状图：收敛后最后100轮平均值

---

## 阶段六：验证与对比

### Step 13 — 端到端验证
**验收标准：**
- [ ] AMAPPO训练曲线可见收敛趋势（奖励上升并趋于稳定）
- [ ] AMAPPO系统成本低于MAPPO（验证异步机制优势）
- [ ] 约束违反率随训练降低

---

## 依赖安装

```bash
pip install torch>=2.0.0 torch-geometric>=2.3.0 numpy>=1.21.0 scipy>=1.7.0
pip install pandas>=1.3.0 matplotlib>=3.4.0 seaborn>=0.11.0
pip install tensorboard>=2.8.0 gymnasium>=0.26.0 networkx>=2.6.0
```

---

## 开发顺序总览

```
Step 1  dag_generator.py     ← 基础数据结构
Step 2  channel_model.py     ← 物理层公式
Step 3  sec_env.py           ← 完整Gym环境
Step 4  gnn_encoder.py       ← 图神经网络
Step 5  actor.py + critic.py ← 策略/价值网络
Step 6  agent.py             ← 智能体封装
Step 7  buffer.py            ← 经验回放
Step 8  config.py + logger.py← 工具支撑
Step 9  mappo.py             ← 同步基线
Step 10 amappo.py            ← 异步核心
Step 11 train.py             ← 训练入口
Step 12 plot_results.py      ← 结果可视化
Step 13 端到端验证            ← 成功标准确认
```
