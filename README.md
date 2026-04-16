# AMAPPO — 异步多智能体近端策略优化

面向卫星边缘计算（SEC）场景的异步 MARL 算法复现项目。

---

## 项目背景

本项目复现论文中提出的 **AMAPPO（Asynchronous Multi-Agent Proximal Policy Optimization）** 算法，并与同步基线 **MAPPO** 对比。系统模拟了一个四层边缘计算架构，由多架 UAV 协同完成 IoT 设备的任务卸载决策。

---

## 系统架构

### 四层计算层次

```
IoTD (N=100) → UAV (M=4, agents) → LEO 卫星 (K=8) → 云端 (1)
```

每架 UAV 作为一个独立智能体，负责管理一组 IoT 设备，决策包括：

| 决策维度 | 说明 |
|---|---|
| 卸载目标（4类） | 本地 / UAV / LEO卫星 / 云端 |
| 带宽分配 | 控制上行链路分配比例 |
| 算力分配 | 控制目标服务器分配比例 |
| UAV 位移（2D） | 控制本步骤移动方向与距离 |

### 通信链路模型

| 链路 | 模型 | 带宽 |
|---|---|---|
| G2U / U2G | Rician 衰落（K因子随仰角变化） | 20 MHz |
| G2S / S2G / U2S / S2U | Shadowed-Rician 衰落 | 15 MHz |
| ISL / S2C | 自由空间路径损耗 | 1 GHz |

---

## 代码结构

```
AMAPPO复现/
├── env/                        # 仿真环境模块
│   ├── dag_generator.py        # 随机 DAG 生成器（任务依赖图）
│   ├── channel_model.py        # 物理层信道模型（Rician / Shadowed-Rician / FSPL）
│   └── sec_env.py              # Gym 风格多智能体环境（reset / step 接口）
│
├── models/                     # 神经网络模块
│   ├── gnn_encoder.py          # 双路 GNN 编码器（任务DAG + 资源图）
│   ├── actor.py                # Actor 网络（GRU + 离散/连续混合动作头）
│   ├── critic.py               # Critic 网络（全局状态价值估计）
│   └── agent.py                # Agent 封装（act / get_value / evaluate_actions）
│
├── algorithms/                 # 训练算法
│   ├── mappo.py                # 同步 MAPPO 训练器（基线）
│   └── amappo.py               # 异步 AMAPPO 训练器（双时钟机制）
│
├── utils/                      # 工具模块
│   ├── config.py               # 超参数配置（dataclass）
│   ├── buffer.py               # 经验缓冲区（AgentBuffer / GlobalBuffer / GAE）
│   └── logger.py               # TensorBoard 日志封装
│
└── experiments/                # 实验入口
    ├── train.py                # 训练主入口（命令行参数解析）
    └── plot_results.py         # 结果可视化（收敛曲线 + 对比柱状图）
```

---

## 核心算法

### AMAPPO 双时钟机制

AMAPPO 与 MAPPO 的核心区别在于**异步决策**：

- **全局时钟 `t'`**：主循环驱动，每步所有环境状态推进一格
- **智能体本地时钟 `t_i`**：记录每个智能体"下次可用"的全局时刻

```
每个全局时间槽:
  ├─ 可用智能体 (t' >= t_i): 执行 act() → 写入经验缓冲 → 更新 t_i
  └─ 不可用智能体 (t' < t_i): 沿用上一动作 → 不写入缓冲
```

`t_i` 的增量由任务计算量（Gcycles）决定，复杂任务占用更多时间槽，从而产生各智能体**决策频率不同**的异步行为。

### 网络结构

```
观测 (37维) → GNN编码器 (128维联合嵌入)
                ↓
            Actor (GRU-64)
              ├─ 离散头: softmax(4) → 卸载目标
              └─ 连续头: sigmoid×2 + raw×2 → [带宽, 算力, Δx, Δy]

全局观测 (148维 = 37×4) → Critic (GRU-64) → V(s)
```

### PPO 更新

1. 从 GlobalBuffer 采样 mini-batch（默认 128）
2. 计算 GAE 优势估计（γ=0.99, λ=0.95）
3. 最小化 Critic TD 误差
4. 最大化 Actor PPO-clip 目标（ε=0.2）
5. 梯度裁剪（max_norm=0.5）后更新

### 奖励函数

```
r_i = -η_t · T_i - η_e · E_i - Σ λ · Φ_ι

Φ_1: 任务超时        Φ_2: UAV 碰撞
Φ_3: UAV 飞出区域    Φ_4: UAV 超速
Φ_5: 资源过载        λ = 10, η_t = η_e = 0.5
```

---

## 环境配置

```bash
conda create -n appo python=3.10
conda activate appo

pip install torch>=2.0.0
pip install torch-geometric>=2.3.0
pip install numpy>=1.21.0 scipy>=1.7.0
pip install networkx>=2.6.0
pip install gymnasium>=0.26.0
pip install tensorboard>=2.8.0
pip install matplotlib>=3.4.0 seaborn>=0.11.0 pandas>=1.3.0
```

---

## 启动命令

### 训练

```bash
conda activate appo

# 训练 AMAPPO（默认配置）
python experiments/train.py --algo amappo --epochs 1500 --seed 42

# 训练 MAPPO（同步基线）
python experiments/train.py --algo mappo --epochs 1500 --seed 42

# 训练 AMAPPO（使用 CUDA）
python experiments/train.py --algo amappo --epochs 1500 --seed 42 --device cuda

# 多 seed 对比实验
for seed in 42 123 456 789 1024; do
    python experiments/train.py --algo amappo --epochs 1500 --seed $seed
    python experiments/train.py --algo mappo  --epochs 1500 --seed $seed
done
```

支持覆盖任意配置项：

```bash
python experiments/train.py \
    --algo amappo \
    --epochs 1500 \
    --seed 42 \
    --lr 5e-4 \
    --mini_batch_size 128 \
    --max_steps 200 \
    --device cpu \
    --log_dir runs \
    --checkpoint_dir checkpoints
```

### 查看训练曲线（TensorBoard）

```bash
tensorboard --logdir runs
# 浏览器打开 http://localhost:6006
```

### 可视化结果对比

```bash
# 默认读取 runs/amappo 和 runs/mappo
python experiments/plot_results.py --output_dir figures

# 指定目录
python experiments/plot_results.py \
    --amappo_dir runs/amappo \
    --mappo_dir  runs/mappo \
    --output_dir figures
```

输出图表：
- `figures/convergence_reward.png`：AMAPPO vs MAPPO 奖励收敛曲线（均值 ± 标准差）
- `figures/convergence_cost.png`：系统成本收敛曲线
- `figures/bar_comparison.png`：最后 100 轮平均性能对比柱状图

---

## 超参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| N | 100 | IoT 设备数量 |
| M | 4 | UAV 数量（智能体数） |
| K | 8 | LEO 卫星数量 |
| J | 20 | 每 DAG 任务数 |
| γ | 0.99 | 折扣因子 |
| λ (GAE) | 0.95 | GAE 平滑系数 |
| ε (PPO clip) | 0.2 | PPO 截断范围 |
| α (lr) | 5×10⁻⁴ | 学习率 |
| mini-batch | 128 | 每次更新样本量 |
| GRU hidden | 64 | GRU 隐层维度 |
| max_grad_norm | 0.5 | 梯度裁剪阈值 |
| max_steps | 200 | 每 episode 最大步数 |
| epochs | 1500 | 训练总 episode 数 |

---

## 验收标准

- [x] AMAPPO 训练曲线可见收敛趋势（奖励上升并趋于稳定）
- [x] 各智能体在同一 episode 内决策次数不同（异步机制验证）
- [ ] AMAPPO 系统成本低于 MAPPO（需完整 1500 轮训练验证）
- [ ] 约束违反率随训练降低
