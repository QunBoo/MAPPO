# AMAPPO 资源图细粒度建模 Debug 文档

日期：2026-04-27

## 问题概述

当前项目原本把资源状态压缩成 4 个聚合节点：

- `local`
- `uav`
- `sat`
- `cloud`

这使 `res_x` 固定为 `(4, 2)`，只能表达“类别级”平均资源状态，无法表达每个 UAV 和每颗卫星的独立负载。对 SEC 场景来说，这会直接削弱资源竞争、最近卫星路由、以及多节点异构资源分配的真实性。

## 根因定位

问题不是单点 bug，而是一条完整假设链：

1. `algorithms/amappo_v2.py`
   - `_build_graph_inputs_v2()` 直接硬编码 4 个资源节点。
2. `algorithms/mappo.py`
   - `_build_graph_inputs()` 同样硬编码 4 个资源节点。
3. `env/sec_env.py`
   - `server_loads` 只维护 4 维聚合负载。
   - 观测布局固定为 `37 = 5 + 20 + 4 + 8`。
   - `resource_graph` 只暴露聚合资源。
4. `models/agent.py` / `models/v2/agent_v2.py`
   - 依赖固定切片读取 `server_states` 和 `prev_action`。
5. `models/gnn_encoder.py` / `models/v2/gnn_encoder_v2.py`
   - 注释、测试样例和 shape 假设都围绕 `(4, 2)` / `(4, 64)`。
6. `tests/*` 与规格文档
   - 多处把资源图和 server state 当成固定 4 节点设计。

## 影响面

### 环境层

- 无法记录“当前 agent 的 UAV 节点”和“最近卫星节点”的独立资源状态。
- `sat` 只是一类资源，不是 `sat_k`。
- `uav` 只是一类资源，不是 `uav_m`。

### 观测层

- 单智能体观测中的 `server_states` 只有 4 维聚合量。
- Critic 输入维度被间接固定在 `37 * M`。

### 编码层

- GNN 的资源图编码学习到的是类别级表示，而不是节点级表示。
- `server_embs.mean(dim=0)` 的上游输入本身就丢失了节点粒度。

### 训练层

- v2 和 v1 都共用 `SECEnv`，如果只改 v2、不改 v1，旧训练链路会直接 shape mismatch。

## 修复方案

### 新的资源节点定义

资源图节点数改为：

- `R = M + K + 2`

固定顺序为：

- `local`
- `uav_0 .. uav_{M-1}`
- `sat_0 .. sat_{K-1}`
- `cloud`

### 新的资源状态表达

- `resource_loads.shape == (R,)`
- `res_x.shape == (R, 2)`
- 两列特征为：
  - `current_load`
  - `normalized_capacity`

### 新的路由映射

- offload 到 `local` -> 更新全局 `local` 节点
- offload 到 `UAV` -> 更新当前 agent 的 `uav_m`
- offload 到 `sat` -> 更新 `nearest_sat(m)`
- offload 到 `cloud` -> 更新全局 `cloud`

### 新的观测布局

- `obs_dim = 5 + 20 + R + 8 = 35 + M + K`
- `obs[25:-8]` 为细粒度 `server_states`
- `obs[-8:]` 为上一动作

## 本次实现结果

已完成以下落地修改：

- `Config` 增加派生字段同步：
  - `resource_node_count = M + K + 2`
  - `obs_dim = 35 + M + K`
- `SECEnv` 用 `resource_loads` 取代 `server_loads`
- `SECEnv.get_resource_graph_data()` 统一提供细粒度资源图张量
- `AMAPPOv2` 和 `MAPPO` 的构图入口改为复用环境资源图接口
- `models/agent.py` 与 `models/v2/agent_v2.py` 改为动态切片观测
- `models/actor.py` 改为支持可变 `server_dim`
- 资源图和训练 smoke 测试改成按变量 `R` 验证

## 已补充的验证

仓库内新增或更新的验证覆盖：

- 资源图 shape 与节点顺序
- 细粒度观测维度
- UAV / 最近卫星 / cloud 的负载路由更新
- `AMAPPOv2Trainer._run_episode()` smoke
- `MAPPOTrainer._run_episode()` smoke
- CLI 2-epoch smoke 用例

## 当前环境备注

2026-04-27 本地验证时观察到：

- 默认 `python` 环境缺少 `torch`
- `conda` 环境 `appo` 可导入 `torch` 和 `torch_geometric`
- `conda` 环境 `appo` 当前缺少 `pytest`

因此本次验证主要使用：

- `conda run -n appo python -c ...`
- `conda run -n appo python experiments/train_v2.py ...`
- `conda run -n appo python experiments/train.py ...`
