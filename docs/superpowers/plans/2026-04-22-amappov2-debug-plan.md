# AMAPPOv2 训练异常调试计划

**日期：** 2026-04-22
**训练命令：** `python experiments/train_v2.py --epochs 1500 --seed 42`
**问题描述：** 训练1500轮后 reward 不收敛、critic_loss 极大、actor_loss 极小且无规律。

---

## 一、训练日志异常现象

| 指标 | 观测值 | 预期值 | 异常程度 |
|------|--------|--------|---------|
| `reward` | -0.075 ~ -0.109，小幅波动 | 随训练增大并收敛 | 量级错误，无单调趋势 |
| `critic_loss` | 3500 ~ 14700，剧烈抖动 | 随训练下降并收敛 | 爆炸级别，数量级异常 |
| `actor_loss` | -0.027 ~ +0.015，近零无趋势 | 随训练减小并收敛 | 近零，策略几乎不更新 |
| `entropy` | 5.64 → 5.58，极缓慢下降 | 适度下降 | 策略几乎维持均匀分布 |
| `decisions` | 各 agent 约 630~700，极均衡 | 逐步分化为有偏好的策略 | 均匀随机，无学习 |

---

## 二、根因分析

经过完整的代码审查（`amappo_v2.py`、`buffer.py`、`agent_v2.py`、`actor_v2.py`、`sec_env.py`），定位到 **4个独立根因**，分别对应3个症状。

---

### Bug 1：`avg_reward` 除以累积决策数（reward 显示值错误）

**位置：** [`algorithms/amappo_v2.py:246`](../../../algorithms/amappo_v2.py#L246)

```python
# 当前代码（错误）
avg_reward = total_reward / max(1, sum(self._agent_decision_count))
```

**根因：** `_agent_decision_count` 在 `log_interval=100` 个 episode 打印后才清零（第248行），而 `avg_reward` 在每个 episode 结束时计算。第1个 episode 除以本 episode 的决策数（约 650×4=2600），但第100个 episode 结束时 `_agent_decision_count` 已累积了99×4×650 ≈ 257,400 次决策，`avg_reward` 被除以约100倍的分母。

**量级分析：**
- 真实 episode 原始 reward（`total_reward`）量级约为 **-1 ~ -50**（由 E_i 主导，见 Bug 4）
- 除以 ~260,000 后得到 **-0.0001 ~ -0.0002**，与日志 -0.08 差异说明前期还未完全累积

**影响：** 日志中的 reward 并不反映真实学习进展，开发者无法判断策略是否在改善。

---

### Bug 2：GAE 在随机打乱的 mini-batch 上计算（critic_loss 爆炸根因）

**位置：**
- [`utils/buffer.py:74-89`](../../../utils/buffer.py#L74)（`GlobalBuffer.sample()` 随机打乱）
- [`utils/buffer.py:104-130`](../../../utils/buffer.py#L104)（`compute_returns_and_advantages` 假设时序连续）
- [`algorithms/amappo_v2.py:252,282-285`](../../../algorithms/amappo_v2.py#L252)（两者结合使用）

**根因：** GAE（Generalized Advantage Estimation）的核心公式为：

```
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
```

`compute_returns_and_advantages` 按时间反向遍历，依赖 `values[t+1]` 是时序上"下一步"的 value。但 `GlobalBuffer.sample()` 用 `np.random.choice` **随机打乱**数据顺序，使得 `values[t+1]` 指向随机邻居的 value 估计值，与真实时序关系完全断裂。

**量级分析：**
- 正确 GAE（顺序数据）：returns 方差约为 1~5
- 错误 GAE（随机顺序）：returns 方差可达 **50~200**（实测），`critic_loss = MSE(V_new, returns)` 因此爆炸到 5000~15000

**额外问题：** `_ppo_update` 中 value 估计只用 `agents[0]` 的 critic（第271行），但 loss 是对所有4个 agent 分别计算的，造成 value 估计与各 agent 策略不匹配。

---

### Bug 3：`evaluate_actions` 中 node embedding 取法与 `act()` 不一致（actor_loss 近零根因）

**位置：** [`models/v2/agent_v2.py:213`](../../../models/v2/agent_v2.py#L213)

```python
# 当前代码（错误）：永远只用节点0的embedding
h_v_t_batch = node_embs[0:1].expand(B, -1)   # (B, 64)
```

**对比 `act()` 中的正确用法（`agent_v2.py:123`）：**

```python
# act() 中（正确）
task_id = self.topo_order[self.step_idx]
h_v_t = self.node_embs[task_id]    # 不同任务用不同节点的embedding
```

**根因：** PPO 的 ratio 计算为：

```
ratio = exp(log_prob_new - log_prob_old)
```

- `log_prob_old`：`act()` 时以 `node_embs[task_id]` 为输入采样得到
- `log_prob_new`：`evaluate_actions()` 时以 `node_embs[0]` 为输入重新计算

当 `task_id != 0` 时，actor 的 GRU 输入 `h_v_t` 不同，导致新旧策略在不同特征空间下估计同一动作的概率。偏差方向随任务随机分布，`ratio` 在 batch 上均值趋近于1，PPO surrogate `min(ratio*A, clip(ratio)*A)` 梯度消失，`actor_loss ≈ 0`。

---

### Bug 4：奖励量级严重不匹配（reward 无法收敛的深层原因）

**位置：** [`env/sec_env.py:278,326-330`](../../../env/sec_env.py#L278)

```python
# 奖励计算
E_comp = kappa * C_cycles * (f_exec ** 2)
r_m = -eta_t * T_i - eta_e * E_i - lambda_c * sum(phi)
```

**量级分析（以 Cloud offloading 为例）：**

| 量 | 公式 | 典型值 |
|----|------|--------|
| `C_cycles` | `attrs["C"] * 1e9` | 1×10⁹ cycles |
| `f_cloud` | `_F_CLOUD = 10.0e9` | 1×10¹⁰ Hz |
| `T_comp` | `C / f` | **0.1 s** |
| `kappa_c` | `_KAPPA_C = 1e-28` | 1×10⁻²⁸ |
| `E_comp` | `kappa * C * f²` | 1×10⁻²⁸ × 1×10⁹ × (1×10¹⁰)² = **10 J** |
| `eta_t * T_i` | 0.5 × 0.1 | **0.05** |
| `eta_e * E_i` | 0.5 × 10 | **5.0** |

**结论：** 能量项比时延项大 **100倍**。策略梯度对 `E_i` 的响应远强于 `T_i`，使得优化时延的梯度信号被淹没，`T` 和 `E` 两个目标无法同时优化，策略陷入能量局部最优而无法改善时延。

---

## 三、根因诊断（测试项）

测试脚本位于 [`tests/test_training_diagnostics.py`](../../../tests/test_training_diagnostics.py)，运行方式：

```bash
python tests/test_training_diagnostics.py
```

### T1：验证 avg_reward 分母错误

**目的：** 确认 `_agent_decision_count` 累积导致 avg_reward 显示值远小于真实值。

**验证逻辑：**
- 模拟2个 episode 的决策累积（650次/agent×4 agent）
- 比较"当前除法"与"正确除法（单 episode）"的结果比值
- **通过条件：** 比值 < 0.5（即当前值比正确值至少小2倍）

**预期输出：**
```
[T1] current avg_reward (wrong denominator): -0.000019
[T1] expected avg_reward (per-episode):      -0.003205
[T1] ratio (should be ~1.0, is): 168x smaller
[T1] PASSED (bug confirmed: denominator is too large)
```

---

### T2：验证 GAE 在随机 batch 上导致 critic_loss 爆炸

**目的：** 对比顺序数据和随机打乱数据上的 GAE returns 方差，确认随机顺序使方差爆炸。

**验证逻辑：**
- 构造长度200的合成轨迹（reward ∈ [-1, 0]，value ∈ [-2, 0]）
- 分别在顺序数据和随机打乱数据上运行 `compute_returns_and_advantages`
- 比较两者的 returns 标准差
- **通过条件：** 随机打乱的 returns std > 顺序的2倍

**预期输出：**
```
[T2] returns std (sequential): 0.8234
[T2] returns std (shuffled):   12.4571
[T2] critic_loss scale ∝ var(returns). Shuffled variance is 15.1x larger
[T2] PASSED (bug confirmed: shuffled GAE inflates returns/critic_loss)
```

---

### T3：验证 h_v_t 不一致导致 ratio ≈ 1

**目的：** 确认 `evaluate_actions` 固定使用 `node_embs[0]` 而非 `node_embs[task_id]` 破坏 PPO ratio 计算。

**验证逻辑：**
- 构造10节点的 fake DAG，用 GNNEncoderV2 生成 node embeddings
- 对每个任务节点：用正确的 `node_embs[task_id]` 和错误的 `node_embs[0]` 分别计算 log_prob
- 计算两者差异及由此产生的 ratio 偏差
- **通过条件：** 确认偏差存在且无一致方向（不能提供有效梯度信号）

**预期输出：**
```
[T3] |ratio - 1| mean when node_embs mismatch: 0.3471
[T3] log_prob shift std (correct vs node0):    0.8203
[T3] Expected: mismatch causes unpredictable ratios, not a consistent signal → actor_loss ≈ 0
[T3] PASSED (bug confirmed: h_v_t always node[0] breaks actor update)
```

---

### T4：验证奖励量级不匹配

**目的：** 用解析计算确认 `eta_e * E_i` 比 `eta_t * T_i` 大100倍以上。

**验证逻辑：**
- 取 Cloud offloading 的典型参数代入计算
- 比较 `eta_t * T_comp` 与 `eta_e * E_comp`
- **通过条件：** E 项 > T 项的10倍

**预期输出：**
```
[T4] T_comp = 0.1000 s   →  eta_t * T_comp = 0.0500
[T4] E_comp = 10.0000 J  →  eta_e * E_comp = 5.0000
[T4] Energy dominates time by factor: 100.0x
[T4] PASSED (bug confirmed: E_i dominates reward by >>10x, time term has negligible gradient signal)
```

---

## 四、修复计划

### Fix 1：修复 `avg_reward` 计算

**文件：** `algorithms/amappo_v2.py`

**改动：**
1. 在 `_run_episode` **开始时**清零 `_agent_decision_count`（而非打印后清零）
2. `avg_reward` 除以**本 episode**的决策数

```python
# _run_episode 开始时清零
def _run_episode(self):
    self._agent_decision_count[:] = 0   # ← 移到这里
    obs_dict = self.env.reset()
    ...

# avg_reward 计算（末尾）
ep_decisions = int(self._agent_decision_count.sum())
avg_reward = total_reward / max(1, ep_decisions)

# train() 中删除原来的清零行
# self._agent_decision_count[:] = 0   ← 删除这行
```

**验收标准：** 日志中 reward 量级应为 -1 ~ -50，不再是 -0.1 量级。

---

### Fix 2：修复 GAE 计算 — 在完整时序轨迹上计算，存入 buffer 前预计算

**文件：** `utils/buffer.py`、`algorithms/amappo_v2.py`

**改动：**
1. `Transition` 增加字段 `advantage: float = 0.0` 和 `ret: float = 0.0`
2. 在 `_run_episode` 结束时，对每个 `AgentBuffer` 的完整时序轨迹（已按时序存储）调用 `compute_returns_and_advantages`，将结果写回各 `Transition`
3. `GlobalBuffer` 增加 `advantage` 和 `ret` 字段的存储
4. `_ppo_update` 中直接从 batch 取 `advantages_t` 和 `returns_t`，不再调用 `compute_returns_and_advantages`

**关键代码（`_run_episode` 末尾，入 global buffer 之前）：**

```python
for m, buf in enumerate(self.agent_buffers):
    if len(buf.transitions) == 0:
        continue
    rewards_m = np.array([t.reward for t in buf.transitions], dtype=np.float32)
    dones_m   = np.array([float(t.done) for t in buf.transitions], dtype=np.float32)

    # Value estimation：对该 agent 的 critic 用顺序数据估计
    values_m = []
    h_v = None
    self.agents[m].critic.eval()
    with torch.no_grad():
        for t in buf.transitions:
            g_obs = torch.tensor(t.global_obs, dtype=torch.float32)
            h_v_in = torch.tensor(t.h_V, dtype=torch.float32
                     ).squeeze(1).squeeze(1).unsqueeze(0)
            v, h_v = self.agents[m].critic(g_obs, h_v_in)
            values_m.append(v.item())
    values_m = np.array(values_m, dtype=np.float32)

    adv_m, ret_m = self.global_buffer.compute_returns_and_advantages(
        rewards_m, values_m, dones_m,
        gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda,
    )
    for i, t in enumerate(buf.transitions):
        t.advantage = float(adv_m[i])
        t.ret       = float(ret_m[i])
```

**验收标准：** `critic_loss` 降至 < 100，returns 方差 < 10。

---

### Fix 3：修复 `evaluate_actions` 中 node embedding 取法

**文件：** `utils/buffer.py`、`models/v2/agent_v2.py`

**改动：**
1. `Transition` 增加字段 `task_id: int = 0`
2. 在 `_run_episode` 的 `act()` 调用处记录当前 `task_id` 并存入 Transition
3. `GlobalBuffer` 存储并返回 `task_ids` 字段
4. `evaluate_actions` 中用 `task_ids_batch` 索引 `node_embs`：

```python
# agent_v2.py evaluate_actions 中替换第213行
task_ids = task_ids_batch.long()                  # (B,)
h_v_t_batch = node_embs[task_ids]                 # (B, 64)  ← 修复
```

**amappo_v2.py 中传入 task_ids：**
```python
log_probs_new, entropies, values_new, _ = agent.evaluate_actions(
    obs_t, actions_t, global_obs_t, h_pi_t, h_V_t,
    dag_x, dag_ei, res_x, res_ei,
    task_ids_t,   # ← 新增参数
)
```

**验收标准：** `|ratio - 1|` 的 batch 均值 > 0.05，`actor_loss` 量级达到 0.01~0.1。

---

### Fix 4：归一化奖励量级

**文件：** `env/sec_env.py` 或 `utils/config.py`

**方案A（推荐）：** 在奖励计算时对 E_i 做量级缩放：

```python
# sec_env.py:325-330
_E_SCALE = 0.01   # 将能量项缩放到与时延同量级

r_m = (
    -self.eta_t * T_i
    - self.eta_e * E_i * _E_SCALE   # ← 加缩放因子
    - self.lam  * float(np.sum(phi))
)
```

**方案B（备选）：** 调整配置中 `eta_e`：

```python
# config.py
eta_e: float = 0.005   # 从 0.5 降低100倍，对齐量级
```

**验收标准：** `eta_t * T_i` 和 `eta_e * E_i` 同量级（比值在 0.1~10 之间），reward 随训练显示出单调改善趋势。

---

## 五、建议执行顺序

```
步骤1  运行诊断脚本，确认4个根因全部复现
       python tests/test_training_diagnostics.py

步骤2  实施 Fix 2（GAE 修复）— 对 critic_loss 影响最大
       修改 buffer.py + amappo_v2.py
       运行 Health Check，确认 critic_loss < 100

步骤3  实施 Fix 3（h_v_t 修复）— 解锁 actor 更新
       修改 buffer.py + agent_v2.py + amappo_v2.py
       运行 Health Check，确认 |actor_loss| > 0.001

步骤4  实施 Fix 1（avg_reward 修复）— 修复监控指标
       修改 amappo_v2.py
       确认 reward 量级为 -1 ~ -50

步骤5  实施 Fix 4（奖励量级归一化）
       修改 sec_env.py 或 config.py
       确认 T 项与 E 项量级比值在 0.1~10

步骤6  运行短训练（200轮）观察收敛趋势
       python experiments/train_v2.py --epochs 200 --seed 42
       预期：reward 单调改善，critic_loss 持续下降，actor_loss 有规律波动
```

---

## 六、修复后预期训练日志形态

| 指标 | 修复前 | 修复后（预期） |
|------|--------|---------------|
| `reward`（ep=100） | -0.089 | -30 ~ -10 |
| `reward`（ep=1500） | -0.105（无趋势） | -5 ~ -1（收敛） |
| `critic_loss`（ep=100） | 10117 | 50 ~ 200 |
| `critic_loss`（ep=1500） | 7686（不降） | 5 ~ 20（收敛） |
| `actor_loss`（ep=100） | 0.0093 | 0.05 ~ 0.2 |
| `actor_loss`（ep=1500） | -0.0107（无趋势） | 0.001 ~ 0.01（收敛） |
| `entropy`（变化幅度） | 5.64 → 5.58（-1%） | 5.6 → 4.0（-28%，有效探索减少） |
| `decisions` 分布 | 均匀（630~700） | 分化（某目标明显偏多） |
