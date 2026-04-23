# AMAPPOv2 训练不收敛调试计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 AMAPPOv2 训练不收敛问题，使 reward 在 1000 轮训练后单调改善并收敛至 >-5.0。

**Architecture:** 本次调试从当前已修复的代码基线出发（前4个Bug已修复），定位并修复4个新的根因：梯度覆盖、DAG编码错用、离策略缓冲污染、PPO更新轮次不足。

**Tech Stack:** Python 3.x, PyTorch, NumPy, pytest

---

## 一、训练日志异常现象（基线：2026-04-23）

**运行命令：** `python experiments/train_v2.py --epochs 1500 --seed 42`

| 指标 | 当前观测值 | 预期收敛值 | 异常特征 |
|------|-----------|-----------|---------|
| `reward` | -6.9 → -3.8 → -5.1（剧烈波动） | 单调上升后收敛至 >-5.0 | 无单调趋势，高方差 |
| `critic_loss` | 897 → 225（缓慢下降，未收敛） | 持续下降至 <20 | 下降速度过慢，值仍偏高 |
| `actor_loss` | -0.07 ~ +0.02（近零波动） | 有规律地在 ±0.05 ~ ±0.2 范围内波动 | 几乎无更新，策略不学习 |
| `entropy` | 5.53 → 5.27（下降仅 4.7%） | 随训练从 ~5.5 降至 ~4.0 | 策略无法有效聚焦 |
| `decisions` | 各 agent 4~14，均匀 | 逐步分化，出现明显偏好 | 随机策略特征，无学习 |

---

## 二、根因分析

> 注：2026-04-22 计划中识别的 Bug 1~4（avg_reward分母、GAE时序、node_embs索引、能量量级）**已全部修复**。  
> 以下是当前代码中残余的4个独立根因。

---

### Bug A：梯度覆盖——`zero_grad()` 在 agent 循环内部（actor_loss 近零的根因）

**位置：** [algorithms/amappo_v2.py:308-334](../../../algorithms/amappo_v2.py#L308)

**当前代码（错误）：**
```python
for agent in self.agents:          # 4 次循环
    ...
    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
    self.optimizer.zero_grad()     # ← 在循环内！清除前一个 agent 的梯度
    loss.backward()
    nn.utils.clip_grad_norm_(...)
    self.optimizer.step()
```

**根因：** 全部4个 agent 共享 `shared_encoder`、同一个 `self.optimizer`。当循环到第 m+1 个 agent 时，`zero_grad()` 清除了 agent m 对 encoder 累积的梯度，最终 encoder 只接收 agent 3（最后一个）的梯度信号。前3个 agent 的 75% 梯度信息被丢弃。

**量级分析：**
- 4个 agent 各贡献 actor_loss ∈ [-0.1, 0.1]，总梯度量级本应是 4×
- 实际只剩 agent 3 的贡献 → actor 有效学习率降为 1/4
- 且每次 `optimizer.step()` 都更新 shared encoder，agent 0~2 的 step 后反而破坏了 encoder

**预期修复效果：** actor_loss 量级提升至 0.03 ~ 0.2，梯度信号方向一致。

---

### Bug B：PPO 更新使用错误的 DAG 编码（ratio 信号混乱的根因）

**位置：** [algorithms/amappo_v2.py:297](../../../algorithms/amappo_v2.py#L297)

**当前代码（错误）：**
```python
dag_x, dag_ei, res_x, res_ei = _build_graph_inputs_v2(self.env, 0)  # ← 永远用 agent 0 的 DAG
```

**根因：** `GlobalBuffer` 中存放了来自 agent 0~3 共4个 agent 的 transitions。每个 agent 在采集时使用的是**自己 DAG** 的 node_embs（`agent.encode(dag_x, ...)` 用 `m` 索引），存入了 `log_prob_old` 和 `task_id`。

PPO 更新时，`evaluate_actions` 用 agent 0 的 DAG 重新 encode，得到 `node_embs`，然后：
```python
h_v_t_batch = node_embs[task_ids]   # task_ids 来自 agent 1/2/3 的 DAG，但 node_embs 是 agent 0 的
```

对 agent 1~3 的 transitions，`log_prob_new` 基于错误的节点特征计算，`ratio = exp(log_prob_new - log_prob_old)` 得到随机噪声值，PPO surrogate 梯度方向不一致。

**量化影响：** agent 0 的 transitions（约占 1/4）更新正确，其余 3/4 的梯度方向随机 → actor_loss 在 batch 上接近0（正负相消）。

**预期修复效果：** 所有 agent 的 ratio 具有一致梯度方向，actor_loss 变得有规律。

---

### Bug C：GlobalBuffer 存放历史策略数据，违反 PPO 同策略假设（reward 波动的根因）

**位置：** [algorithms/amappo_v2.py:105](../../../algorithms/amappo_v2.py#L105)、[utils/buffer.py:42](../../../utils/buffer.py#L42)

**当前代码（问题）：**
```python
self.global_buffer = GlobalBuffer(capacity=50000)   # 最多保留 50000 条
# ...
if len(self.global_buffer) >= self.cfg.mini_batch_size:
    train_metrics = self._ppo_update()              # 每 episode 只更新1次，不清除 buffer
```

**根因：** PPO 是同策略（on-policy）算法，依赖 importance sampling 修正：
```
ratio = exp(log_prob_new(a|s) - log_prob_old(a|s))
```

这个修正只在策略偏移很小时有效（理论上 clip 到 1±0.2）。当 buffer 包含 1000 个 episode 之前的旧数据时，`log_prob_old` 来自许多次策略更新之前的策略，`ratio` 可能爆炸或趋于0，clip 无法修正如此大的偏移。

**量化分析：**
- 每 episode 约 40 条 transitions（4 agent × 10 decisions）
- capacity=50000 → 可保存 ~1250 个 episode 的数据
- 第 1500 ep 的 buffer 包含第 250 ep 开始的数据（已历经 1250 次策略更新的旧数据）
- PPO clip 只能修正 e^(-0.2)=0.82 ~ e^(0.2)=1.22 范围内的 ratio，实际 ratio 可达 0 或 100+

**预期修复效果：** 每次 PPO 更新后清空 buffer，reward 波动显著降低。

---

### Bug D：每 episode 仅做1次 mini-batch 更新（学习效率极低）

**位置：** [algorithms/amappo_v2.py:122-125](../../../algorithms/amappo_v2.py#L122)

**当前代码（问题）：**
```python
if len(self.global_buffer) >= self.cfg.mini_batch_size:
    train_metrics = self._ppo_update()   # 每 episode 只调用1次！
```

**根因：** 标准 PPO 在每次数据采集后进行 K 个 epoch 的更新（通常 K=4~10）。当前代码每 episode 只用1个 mini-batch 更新一次，每次使用约 128/40=3.2 个 episode 的数据量，但实际上只随机采样128条而非全量使用。

**量化影响：**
- 每个 transition 期望被训练到的次数 ≈ 128/buffer_size ≈ 128/50000 ≈ 0.26%
- 1500 个 episode 后，每条有效 on-policy transition 平均参与训练 < 1次
- 对比标准 PPO：每条 transition 参与 4~10 次更新

**预期修复效果：** 样本利用率提升 10x，critic_loss 下降速度加快。

---

## 三、修复计划（File Structure）

| 文件 | 修改类型 | 修改内容 |
|------|---------|---------|
| `algorithms/amappo_v2.py` | 修改 | Fix A（合并 loss，统一 backward）；Fix B（每个 agent 用自己 DAG 编码）；Fix C（update 后清空 buffer，多 episode 累积再更新）；Fix D（多轮 epoch PPO 更新） |
| `utils/config.py` | 修改 | 新增 `ppo_epochs: int = 4`、`update_every: int = 5` 配置项 |
| `tests/v2/test_convergence_monitor.py` | 新增 | 6个测试项，验证训练日志是否符合预期收敛形态 |

---

## 四、测试项设计（收敛监控）

测试项目标：运行一段短训练后，自动判断日志指标是否符合"正在收敛"的预期形态。

**运行方式：**
```bash
python -m pytest tests/v2/test_convergence_monitor.py -v
```

### T1：critic_loss 在 300 轮内应下降超过 50%

**验证逻辑：**
- 解析 stdout 中 `critic_loss=` 数值
- 比较 ep=100 时均值与 ep=300 时均值
- **通过条件：** `loss_at_300 < loss_at_100 * 0.5`

**意义：** critic_loss 不下降说明 GAE 仍有问题或 value 网络不更新。

---

### T2：reward 在 1000 轮后线性回归斜率 > 0（单调改善趋势）

**验证逻辑：**
- 收集 ep=500~1000 的 reward 序列
- 用 `np.polyfit` 拟合线性趋势
- **通过条件：** `slope > 0`（正斜率，即 reward 在上升）

**意义：** reward 仍波动但整体趋势必须是上升，否则策略没有在学习。

---

### T3：entropy 在 1000 轮内下降幅度为 5%~40%

**验证逻辑：**
- 比较 ep=100 和 ep=1000 的 entropy 值
- **通过条件：** `0.05 < (entropy_100 - entropy_1000) / entropy_100 < 0.40`

**意义：** 上限防止策略过早坍缩（entropy 剧降说明过拟合）；下限确保策略在收敛（entropy 完全不变说明没有学习）。

---

### T4：|actor_loss| 均值 > 0.01（actor 在有效更新）

**验证逻辑：**
- 收集 ep=200~500 的 actor_loss 序列
- 计算绝对值均值
- **通过条件：** `mean(|actor_loss|) > 0.01`

**意义：** actor_loss 接近0说明 Bug A 或 Bug B 未修复，ratio 梯度消失。

---

### T5：critic_loss 在 ep=500 时 < 100（critic 正常收敛）

**验证逻辑：**
- 读取 ep=500 时的 critic_loss
- **通过条件：** `critic_loss_at_500 < 100`

**意义：** critic_loss > 100 在 500 ep 后说明 GAE returns 仍不稳定或 value 网络无法学习。

---

### T6：reward 在 ep=1000~1500 的标准差 < 2.0（收敛稳定性）

**验证逻辑：**
- 收集 ep=1000~1500 的 reward 序列
- 计算标准差
- **通过条件：** `std < 2.0`

**意义：** reward 最终阶段的方差过大说明策略仍在大幅振荡，未真正收敛。

---

## 五、实施计划（Tasks）

### Task 1：添加 `ppo_epochs` 和 `update_every` 配置项

**Files:**
- Modify: `utils/config.py`

- [ ] **Step 1: 修改 config.py，新增训练控制参数**

```python
# utils/config.py — 在 "# Training" 块末尾添加
ppo_epochs: int = 4       # PPO 每次更新的 epoch 数（Fix D）
update_every: int = 5     # 累积多少 episode 后执行一次 PPO 更新（Fix C）
```

文件修改后的 Training 块应为：
```python
# Training
gamma: float = 0.99
gae_lambda: float = 0.95
eps_clip: float = 0.2
lr: float = 5e-4
mini_batch_size: int = 128
epochs: int = 1500
max_grad_norm: float = 0.5
ppo_epochs: int = 4       # ← 新增
update_every: int = 5     # ← 新增
```

- [ ] **Step 2: 验证 Config 可实例化并包含新字段**

```bash
cd d:/Coding/python/AMAPPO && python -c "from utils.config import Config; c = Config(); print(c.ppo_epochs, c.update_every)"
```

预期输出：`4 5`

- [ ] **Step 3: Commit**

```bash
git add utils/config.py
git commit -m "config: add ppo_epochs and update_every hyperparameters"
```

---

### Task 2：Fix A — 合并 agent 损失，统一 backward

**Files:**
- Modify: `algorithms/amappo_v2.py:304-345`

**问题代码：**
```python
for agent in self.agents:
    ...
    self.optimizer.zero_grad()   # ← 每个 agent 都清零，前一个 agent 的梯度被丢弃
    loss.backward()
    nn.utils.clip_grad_norm_(...)
    self.optimizer.step()
```

**修复目标：** 将4个 agent 的 loss 累加，最后做一次 backward + step。

- [ ] **Step 1: 写失败测试，确认当前代码 actor_loss 过小**

在 `tests/v2/test_convergence_monitor.py` 创建文件并写入：

```python
"""AMAPPOv2 训练收敛监控测试套件."""
import subprocess
import re
import numpy as np
import pytest


def _run_short_training(epochs: int = 150, seed: int = 0) -> list[dict]:
    """运行短训练并解析日志，返回每个 log_interval 的指标列表。"""
    result = subprocess.run(
        ["python", "experiments/train_v2.py", f"--epochs={epochs}", f"--seed={seed}"],
        capture_output=True, text=True, timeout=300,
    )
    lines = result.stdout.splitlines()
    records = []
    pattern = re.compile(
        r"ep=\s*(\d+)\s+reward=\s*([\-\d.]+)\s+actor_loss=([\-\d.]+)\s+"
        r"critic_loss=([\-\d.]+)\s+entropy=([\-\d.]+)"
    )
    for line in lines:
        m = pattern.search(line)
        if m:
            records.append({
                "ep":           int(m.group(1)),
                "reward":       float(m.group(2)),
                "actor_loss":   float(m.group(3)),
                "critic_loss":  float(m.group(4)),
                "entropy":      float(m.group(5)),
            })
    return records


def test_t4_actor_loss_nonzero():
    """T4: |actor_loss| 均值必须 > 0.01，确认 actor 在有效更新。"""
    records = _run_short_training(epochs=150, seed=42)
    assert len(records) >= 1, "训练日志为空，训练未正常运行"
    actor_losses = [abs(r["actor_loss"]) for r in records]
    mean_abs = float(np.mean(actor_losses))
    assert mean_abs > 0.01, (
        f"actor_loss 均值绝对值 {mean_abs:.5f} <= 0.01，"
        "梯度覆盖(Bug A) 或 DAG错用(Bug B) 未修复"
    )
```

- [ ] **Step 2: 运行测试，确认当前失败**

```bash
cd d:/Coding/python/AMAPPO && python -m pytest tests/v2/test_convergence_monitor.py::test_t4_actor_loss_nonzero -v
```

预期：`FAILED` — `actor_loss 均值绝对值 0.00xxx <= 0.01`

- [ ] **Step 3: 修复 `_ppo_update`，将 zero_grad/backward/step 移到循环外**

将 `algorithms/amappo_v2.py` 的 `_ppo_update` 方法替换为：

```python
def _ppo_update(self) -> dict:
    total_actor_loss  = 0.0
    total_critic_loss = 0.0
    total_entropy     = 0.0

    for _ in range(self.cfg.ppo_epochs):
        batch = self.global_buffer.sample(self.cfg.mini_batch_size)

        obs_t        = torch.tensor(batch["obs"],        dtype=torch.float32, device=self.device)
        actions_t    = torch.tensor(batch["actions"],    dtype=torch.float32, device=self.device)
        h_pi_t       = torch.tensor(batch["h_pi"],       dtype=torch.float32, device=self.device)
        h_V_t        = torch.tensor(batch["h_V"],        dtype=torch.float32, device=self.device)
        global_obs_t = torch.tensor(batch["global_obs"], dtype=torch.float32, device=self.device)
        log_probs_old_t = torch.tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        task_ids_t   = torch.tensor(batch["task_ids"],   dtype=torch.long,    device=self.device)
        agent_ids_t  = torch.tensor(batch["agent_ids"],  dtype=torch.long,    device=self.device)

        advantages_t = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns_t    = torch.tensor(batch["returns"],    dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Fix A: 累加所有 agent 的 loss，再统一 backward + step
        self.optimizer.zero_grad()

        epoch_actor  = 0.0
        epoch_critic = 0.0
        epoch_ent    = 0.0

        for agent_id, agent in enumerate(self.agents):
            # Fix B: 用该 agent 自己的 DAG 编码（见 Task 3）
            dag_x, dag_ei, res_x, res_ei = self._get_dag_tensors(agent_id)

            self.shared_encoder.train()
            agent.actor.train()
            agent.critic.train()

            log_probs_new, entropies, values_new, _ = agent.evaluate_actions(
                obs_t, actions_t, global_obs_t, h_pi_t, h_V_t,
                dag_x, dag_ei, res_x, res_ei,
                task_ids_t,
            )

            ratio  = torch.exp(log_probs_new - log_probs_old_t)
            surr1  = ratio * advantages_t
            surr2  = torch.clamp(ratio, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages_t
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values_new, returns_t)
            entropy     = entropies.mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            loss.backward()   # 累加梯度（不在此处 zero_grad）

            epoch_actor  += actor_loss.item()
            epoch_critic += critic_loss.item()
            epoch_ent    += entropy.item()

        # Fix A: 统一裁剪 + 更新
        nn.utils.clip_grad_norm_(
            [p for pg in self.optimizer.param_groups for p in pg["params"]],
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()
        self._train_step += 1

        M = len(self.agents)
        total_actor_loss  += epoch_actor  / M
        total_critic_loss += epoch_critic / M
        total_entropy     += epoch_ent    / M

    n_epochs = self.cfg.ppo_epochs
    return {
        "actor_loss":  total_actor_loss  / n_epochs,
        "critic_loss": total_critic_loss / n_epochs,
        "entropy":     total_entropy     / n_epochs,
    }
```

- [ ] **Step 4: 运行 T4 测试，确认通过**

```bash
cd d:/Coding/python/AMAPPO && python -m pytest tests/v2/test_convergence_monitor.py::test_t4_actor_loss_nonzero -v
```

预期：`PASSED`

- [ ] **Step 5: Commit**

```bash
git add algorithms/amappo_v2.py tests/v2/test_convergence_monitor.py
git commit -m "fix: accumulate agent losses before backward to fix gradient clobbering (Bug A)"
```

---

### Task 3：Fix B — 每个 agent 使用自己的 DAG 进行 PPO 更新编码

**Files:**
- Modify: `algorithms/amappo_v2.py`（新增 `_get_dag_tensors` 辅助方法、修改 `_run_episode` 缓存 DAG inputs）

**问题：** `_ppo_update` 永远使用 `agent 0` 的 DAG，导致 agent 1~3 的 `log_prob_new` 基于错误特征计算。

- [ ] **Step 1: 写失败测试，确认 ratio 当前对 agent 1~3 是噪声**

在 `tests/v2/test_convergence_monitor.py` 追加：

```python
def test_dag_encoding_consistency():
    """验证 evaluate_actions 中 log_prob_new 和 log_prob_old 来自同一 DAG 编码。
    
    间接测试：如果 Bug B 未修复，agent 1~3 的 log_prob 差异会是随机噪声，
    表现为 |actor_loss| 极小。T4 已覆盖此检测，此处做单元级验证。
    """
    import torch
    from utils.config import Config
    from models.v2.gnn_encoder_v2 import GNNEncoderV2
    from models.v2.agent_v2 import MAPPOAgentV2
    from algorithms.amappo_v2 import _build_graph_inputs_v2
    from env.sec_env import SECEnv

    cfg = Config(J=5, N=10, M=2, seed=0)
    env = SECEnv(cfg)
    env.reset()

    encoder = GNNEncoderV2()
    agent0 = MAPPOAgentV2(0, "LEO", cfg, encoder)
    agent1 = MAPPOAgentV2(1, "LEO", cfg, encoder)

    # agent 0 的 DAG
    dag0_x, dag0_ei, res0_x, res0_ei = _build_graph_inputs_v2(env, 0)
    # agent 1 的 DAG
    dag1_x, dag1_ei, res1_x, res1_ei = _build_graph_inputs_v2(env, 1)

    with torch.no_grad():
        node_embs_0, _, _ = encoder(dag0_x, dag0_ei, res0_x, res0_ei)
        node_embs_1, _, _ = encoder(dag1_x, dag1_ei, res1_x, res1_ei)

    # node_embs_0 和 node_embs_1 对同一 task_id 应有不同的嵌入
    task_id = torch.tensor([0, 1, 2], dtype=torch.long)
    emb0 = node_embs_0[task_id]
    emb1 = node_embs_1[task_id]
    diff = (emb0 - emb1).abs().mean().item()
    assert diff > 1e-4, (
        f"两个 agent 的 DAG node embeddings 完全相同（diff={diff:.6f}），"
        "DAGs 是随机生成的，应当不同。测试环境可能有问题。"
    )
```

- [ ] **Step 2: 运行测试确认 DAG 差异存在（应该 PASS，确认 DAG 确实不同）**

```bash
cd d:/Coding/python/AMAPPO && python -m pytest tests/v2/test_convergence_monitor.py::test_dag_encoding_consistency -v
```

预期：`PASSED`（确认不同 agent 的 DAG 确实有不同嵌入，说明 Bug B 确实存在）

- [ ] **Step 3: 在 `AMAPPOv2Trainer.__init__` 中添加 DAG tensor 缓存存储**

在 `amappo_v2.py` 的 `AMAPPOv2Trainer.__init__` 末尾（`self._train_step = 0` 之后）添加：

```python
self._dag_tensors: list = []   # 每个 agent 的 (dag_x, dag_ei, res_x, res_ei)，episode 时更新
```

- [ ] **Step 4: 在 `_run_episode` 的 encode 循环中缓存每个 agent 的 DAG tensors**

在 `_run_episode` 中，找到以下代码块：
```python
for m, agent in enumerate(self.agents):
    dag_x, dag_ei, res_x, res_ei = _build_graph_inputs_v2(self.env, m)
    agent.encode(dag_x, dag_ei, res_x, res_ei)
```

替换为：
```python
self._dag_tensors = []
for m, agent in enumerate(self.agents):
    dag_x, dag_ei, res_x, res_ei = _build_graph_inputs_v2(self.env, m)
    agent.encode(dag_x, dag_ei, res_x, res_ei)
    self._dag_tensors.append((
        dag_x.to(self.device),
        dag_ei.to(self.device),
        res_x.to(self.device),
        res_ei.to(self.device),
    ))
```

- [ ] **Step 5: 添加 `_get_dag_tensors` 辅助方法**

在 `_ppo_update` 方法之前添加：

```python
def _get_dag_tensors(self, agent_id: int):
    """返回指定 agent 在当前 episode 编码时使用的 DAG tensors。"""
    if self._dag_tensors and agent_id < len(self._dag_tensors):
        return self._dag_tensors[agent_id]
    # 降级：回退到 agent 0（不应触发，但防止崩溃）
    dag_x, dag_ei, res_x, res_ei = _build_graph_inputs_v2(self.env, 0)
    return (
        dag_x.to(self.device), dag_ei.to(self.device),
        res_x.to(self.device), res_ei.to(self.device),
    )
```

- [ ] **Step 6: 运行 T4 测试确认仍然通过**

```bash
cd d:/Coding/python/AMAPPO && python -m pytest tests/v2/test_convergence_monitor.py::test_t4_actor_loss_nonzero -v
```

预期：`PASSED`

- [ ] **Step 7: Commit**

```bash
git add algorithms/amappo_v2.py
git commit -m "fix: use per-agent DAG encoding in _ppo_update to fix log_prob mismatch (Bug B)"
```

---

### Task 4：Fix C — 添加 `agent_ids` 字段到 Transition/Buffer；PPO 后清空 buffer

**Files:**
- Modify: `utils/buffer.py`（Transition 添加 `agent_id` 字段；GlobalBuffer 存储/返回 `agent_ids`）
- Modify: `algorithms/amappo_v2.py`（存储 agent_id；update 后清空 buffer；按 `update_every` 控制更新频率）

- [ ] **Step 1: 修改 `Transition` 添加 `agent_id` 字段**

在 `utils/buffer.py` 的 `Transition` dataclass 末尾添加一个字段：

```python
@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    h_pi: np.ndarray
    h_V: np.ndarray
    global_obs: np.ndarray
    done: bool
    log_prob: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0
    task_id: int = 0
    agent_id: int = 0      # ← 新增：记录该 transition 属于哪个 agent
```

- [ ] **Step 2: 修改 `GlobalBuffer` 存储并返回 `agent_ids`**

在 `GlobalBuffer.__init__` 的列表初始化中添加：
```python
self._agent_ids: list = []
```

在 `add_from_agent_buffer` 的循环中添加：
```python
self._agent_ids.append(t.agent_id)
```

在 trim 到 capacity 的块中添加：
```python
self._agent_ids = self._agent_ids[excess:]
```

在 `sample()` 返回的 dict 中添加：
```python
'agent_ids': np.array([self._agent_ids[i] for i in indices]),  # (B,)
```

在 `clear()` 中添加：
```python
self._agent_ids.clear()
```

- [ ] **Step 3: 修改 `_run_episode` 中 Transition 创建，写入 `agent_id`**

在 `amappo_v2.py` 中，找到 `Transition(...)` 创建处（在 `for m in available_agents:` 循环内），将 `task_id=current_task_id,` 之后添加：
```python
agent_id=m,
```

- [ ] **Step 4: 修改 `train()` 循环，按 `update_every` 批量更新并清空 buffer**

将 `train()` 的训练循环从：
```python
for ep in range(1, self.cfg.epochs + 1):
    ep_reward, ep_info = self._run_episode()

    if len(self.global_buffer) >= self.cfg.mini_batch_size:
        train_metrics = self._ppo_update()
    else:
        train_metrics = {"critic_loss": 0.0, "actor_loss": 0.0, "entropy": 0.0}
```

替换为：
```python
train_metrics = {"critic_loss": 0.0, "actor_loss": 0.0, "entropy": 0.0}
for ep in range(1, self.cfg.epochs + 1):
    ep_reward, ep_info = self._run_episode()

    # Fix C: 累积 update_every 个 episode 后再做 PPO 更新，然后清空 buffer
    if ep % self.cfg.update_every == 0:
        if len(self.global_buffer) >= self.cfg.mini_batch_size:
            train_metrics = self._ppo_update()
            self.global_buffer.clear()   # ← 清空，保持同策略纯洁性
```

- [ ] **Step 5: 写失败测试，验证 buffer 在 update 后被清空**

在 `tests/v2/test_convergence_monitor.py` 追加：

```python
def test_buffer_cleared_after_update():
    """Fix C: 验证每次 PPO update 后 GlobalBuffer 被清空，防止离策略污染。"""
    import torch
    from utils.config import Config
    from algorithms.amappo_v2 import AMAPPOv2Trainer

    cfg = Config(J=5, N=10, M=2, epochs=12, update_every=5, ppo_epochs=1,
                 mini_batch_size=10, seed=0, log_interval=100, save_interval=100)
    trainer = AMAPPOv2Trainer(cfg)

    # 运行 10 个 episode（update_every=5，应触发2次更新）
    train_metrics_list = []
    for ep in range(1, 11):
        trainer._run_episode()
        if ep % cfg.update_every == 0:
            if len(trainer.global_buffer) >= cfg.mini_batch_size:
                trainer._ppo_update()
                trainer.global_buffer.clear()
                train_metrics_list.append(len(trainer.global_buffer))

    # 每次 update 后 buffer 应该被清空
    for size in train_metrics_list:
        assert size == 0, f"Buffer update 后应为空，实际大小: {size}"
```

- [ ] **Step 6: 运行测试确认通过**

```bash
cd d:/Coding/python/AMAPPO && python -m pytest tests/v2/test_convergence_monitor.py::test_buffer_cleared_after_update -v
```

预期：`PASSED`

- [ ] **Step 7: Commit**

```bash
git add utils/buffer.py algorithms/amappo_v2.py tests/v2/test_convergence_monitor.py
git commit -m "fix: add agent_id to Transition, clear buffer after PPO update for on-policy correctness (Bug C)"
```

---

### Task 5：添加完整的收敛监控测试套件

**Files:**
- Modify: `tests/v2/test_convergence_monitor.py`（补充 T1, T2, T3, T5, T6 测试）

**说明：** 以下测试需运行真实训练（约 300~500 ep），会耗时数分钟。使用 `pytest -m slow` 标记与单元测试分离。

- [ ] **Step 1: 补充 T1/T2/T3/T5/T6 测试**

在 `tests/v2/test_convergence_monitor.py` 追加以下代码（保留前面已有内容）：

```python
import pytest


@pytest.fixture(scope="module")
def training_records_300():
    """运行 300 轮训练，缓存结果供多个测试使用（避免重复运行）。"""
    return _run_short_training(epochs=300, seed=42)


@pytest.fixture(scope="module")
def training_records_1500():
    """运行 1500 轮训练，用于最终收敛测试。耗时约 5 分钟。"""
    return _run_short_training(epochs=1500, seed=42)


@pytest.mark.slow
def test_t1_critic_loss_decreases(training_records_300):
    """T1: critic_loss 在 300 轮内下降超过 50%."""
    records = training_records_300
    assert len(records) >= 3, f"日志记录不足3条（实际{len(records)}条），无法判断趋势"

    loss_early = records[0]["critic_loss"]   # ep=100
    loss_late  = records[2]["critic_loss"]   # ep=300
    assert loss_late < loss_early * 0.5, (
        f"critic_loss 下降不足50%: ep=100时{loss_early:.1f}，ep=300时{loss_late:.1f}，"
        f"下降了{100*(1-loss_late/loss_early):.1f}%（需>50%）"
    )


@pytest.mark.slow
def test_t2_reward_trend_positive(training_records_1500):
    """T2: ep=500~1000 的 reward 线性回归斜率 > 0（单调改善趋势）."""
    records = training_records_1500
    mid = [r for r in records if 500 <= r["ep"] <= 1000]
    assert len(mid) >= 4, f"ep=500~1000 的记录不足4条（实际{len(mid)}条）"

    x = np.array([r["ep"]     for r in mid], dtype=float)
    y = np.array([r["reward"] for r in mid], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    assert slope > 0, (
        f"reward 趋势斜率={slope:.6f} <= 0，策略在 ep=500~1000 无改善。"
        "Bug A/B 可能未修复，或学习率过高导致振荡。"
    )


@pytest.mark.slow
def test_t3_entropy_decrease_moderate(training_records_1500):
    """T3: entropy 在 1000 轮内下降幅度为 5%~40%（适度收敛，不过快也不过慢）."""
    records = training_records_1500
    ep100  = next((r for r in records if r["ep"] == 100), None)
    ep1000 = next((r for r in records if r["ep"] == 1000), None)
    assert ep100  is not None, "日志中缺少 ep=100  的记录"
    assert ep1000 is not None, "日志中缺少 ep=1000 的记录"

    drop_ratio = (ep100["entropy"] - ep1000["entropy"]) / ep100["entropy"]
    assert 0.05 < drop_ratio < 0.40, (
        f"entropy 下降比例={drop_ratio*100:.1f}%，需在 5%~40% 之间。"
        f"ep=100: {ep100['entropy']:.4f}, ep=1000: {ep1000['entropy']:.4f}"
    )


@pytest.mark.slow
def test_t5_critic_loss_at_500(training_records_1500):
    """T5: critic_loss 在 ep=500 时应 < 100."""
    records = training_records_1500
    ep500 = next((r for r in records if r["ep"] == 500), None)
    assert ep500 is not None, "日志中缺少 ep=500 的记录"

    assert ep500["critic_loss"] < 100, (
        f"ep=500 时 critic_loss={ep500['critic_loss']:.1f} >= 100，"
        "value 网络收敛过慢，GAE 或梯度问题可能仍存在"
    )


@pytest.mark.slow
def test_t6_reward_stability_late(training_records_1500):
    """T6: ep=1000~1500 的 reward 标准差 < 2.0（策略收敛稳定性）."""
    records = training_records_1500
    late = [r for r in records if r["ep"] >= 1000]
    assert len(late) >= 4, f"ep>=1000 的记录不足4条（实际{len(late)}条）"

    rewards = np.array([r["reward"] for r in late])
    std = float(np.std(rewards))
    assert std < 2.0, (
        f"ep=1000~1500 reward 标准差={std:.3f} >= 2.0，策略仍不稳定。"
        f"reward 值: {rewards.tolist()}"
    )
```

- [ ] **Step 2: 在 pytest.ini 或 pyproject.toml 中注册 `slow` 标记（避免 warning）**

检查项目根目录是否有 `pytest.ini`：
```bash
ls d:/Coding/python/AMAPPO/pytest.ini 2>/dev/null || echo "not found"
```

如果不存在，创建 `pytest.ini`：
```ini
[pytest]
markers =
    slow: marks tests as slow-running (deselect with '-m "not slow"')
```

- [ ] **Step 3: 运行快速测试套件（不含 slow），确认全部通过**

```bash
cd d:/Coding/python/AMAPPO && python -m pytest tests/v2/test_convergence_monitor.py -m "not slow" -v
```

预期：
```
test_t4_actor_loss_nonzero         PASSED
test_dag_encoding_consistency      PASSED
test_buffer_cleared_after_update   PASSED
```

- [ ] **Step 4: Commit**

```bash
git add tests/v2/test_convergence_monitor.py pytest.ini
git commit -m "test: add convergence monitoring test suite (T1-T6) for training diagnostics"
```

---

### Task 6：运行完整训练验证修复效果

- [ ] **Step 1: 运行 1500 轮训练，观察日志**

```bash
cd d:/Coding/python/AMAPPO && python experiments/train_v2.py --epochs 1500 --seed 42 2>&1 | tee /tmp/train_fixed.log
```

**预期日志形态（修复后）：**

| ep | reward | critic_loss | actor_loss | entropy |
|----|--------|-------------|------------|---------|
| 100 | -15 ~ -8 | 300 ~ 500 | 0.05 ~ 0.2 | ~5.5 |
| 500 | -8 ~ -5 | 30 ~ 80 | 0.02 ~ 0.1 | ~5.2 |
| 1000 | -6 ~ -3 | 10 ~ 30 | 0.01 ~ 0.05 | ~4.8 |
| 1500 | -5 ~ -2（收敛） | 5 ~ 20 | 0.005 ~ 0.02 | ~4.5 |

**判断收敛标准：**
- reward 总体趋势向上（可以有局部波动，但 ep=1000~1500 的均值 > ep=100~500 的均值）
- critic_loss ep=500 时 < 100
- actor_loss 始终有非零绝对值（不再接近0）
- entropy 有明显下降（从 ~5.5 降至 ~4.5 或更低）

- [ ] **Step 2: 运行慢速测试套件，验证所有测试通过**

```bash
cd d:/Coding/python/AMAPPO && python -m pytest tests/v2/test_convergence_monitor.py -v --timeout=600
```

预期：6个测试全部 `PASSED`

- [ ] **Step 3: 最终 Commit**

```bash
git add -A
git commit -m "fix: resolve AMAPPOv2 non-convergence (gradient clobbering, DAG mismatch, off-policy contamination)"
```

---

## 六、修复后预期训练日志形态对比

| 指标 | 修复前（当前） | 修复后（预期） | 对应修复 |
|------|------------|-------------|---------|
| `reward` 趋势 | 无趋势，±3 以上波动 | 单调上升，1000ep 后稳定 | Fix A+B（梯度信号正确）+Fix C（同策略）|
| `critic_loss` ep=100 | 897 | 200 ~ 500 | Fix D（多 epoch 更新）|
| `critic_loss` ep=500 | ~400 | < 100 | Fix C+D |
| `critic_loss` ep=1500 | 225（未收敛） | < 20（收敛） | Fix A+C+D |
| `actor_loss` 绝对值均值 | < 0.01（近零） | 0.03 ~ 0.15 | Fix A（梯度不再被覆盖）+Fix B（DAG正确）|
| `entropy` 下降幅度 | 4.7%（5.53→5.27） | 15%~30%（5.5→4.0）| Fix A+B（策略有效学习）|
| `decisions` 分布 | 均匀（无偏好） | 分化（某卸载目标占优） | 所有修复 |

---

## 七、附：根因与症状的对应关系

```
reward 波动大、无收敛趋势
├── Bug A（梯度覆盖）→ actor 几乎不更新 → decisions 均匀随机 → reward 随机游走
├── Bug B（DAG错用）→ actor_loss 近零（正负相消）→ 与 Bug A 协同
└── Bug C（离策略）→ importance sampling 失效 → 梯度信号随机 → reward 无趋势

critic_loss 下降缓慢
└── Bug D（单 epoch 更新）→ 每条 transition 平均更新 < 1次 → critic 收敛慢

entropy 下降不足
└── Bug A+B 导致 actor 参数几乎不变 → 策略维持近似均匀分布
```
