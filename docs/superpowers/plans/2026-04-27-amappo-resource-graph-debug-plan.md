# AMAPPO 资源图细粒度建模 Debug Plan

日期：2026-04-27

## 目标

把资源建模从 4 个聚合节点升级为 `R = M + K + 2` 个细粒度节点，并保证：

- v2 训练链路可运行
- v1 `mappo` 链路不被共享环境改动破坏
- 观测、资源图、测试、规格文档同步收敛

## 已执行的改动

### 1. 配置与环境

- 为 `Config` 增加派生字段同步函数 `sync_derived_fields()`
- 派生：
  - `resource_node_count = M + K + 2`
  - `obs_dim = 5 + 20 + resource_node_count + 8`
- `SECEnv` 新增：
  - `resource_loads`
  - 资源节点索引辅助函数
  - `resource_index_for_offload()`
  - `get_resource_graph_data()`

### 2. 训练链路

- `algorithms/amappo_v2.py`
  - 资源图输入改为调用环境统一接口
- `algorithms/mappo.py`
  - 同步改为调用环境统一接口
- `models/agent.py`
  - 观测切片改为 `obs[25:-8]` 和 `obs[-8:]`
  - `Actor(server_dim=resource_node_count)`
- `models/v2/agent_v2.py`
  - 去掉 `29:37` 固定切片，统一用最后 8 维读取 `a_prev`

### 3. 编码器与测试

- `models/gnn_encoder.py`
  - 资源图注释和示例改为变量 `R`
- `models/v2/gnn_encoder_v2.py`
  - `server_embs` shape 注释改为 `(R, 64)`
- 新增 `tests/test_resource_graph_granularity.py`
  - 资源图 shape / 节点顺序
  - 细粒度观测
  - UAV / SAT / cloud 路由更新
  - v1/v2 episode smoke
  - CLI smoke

## 测试用例

### 资源图与观测

- `test_resource_graph_shapes_and_node_order`
- `test_observation_uses_fine_grained_server_states`
- `test_step_updates_current_uav_node_only`
- `test_step_updates_nearest_satellite_node_only`
- `test_step_updates_cloud_node_only`

### 训练 smoke

- `test_amappov2_run_episode_smoke_with_fine_grained_resources`
- `test_mappo_run_episode_smoke_with_fine_grained_resources`
- `test_amappov2_cli_smoke_with_fine_grained_resources`
- `test_mappo_cli_smoke_with_fine_grained_resources`

### 运行前置检查

- `test_conda_appo_runtime_dependency_probe`

说明：

- 当前 `appo` 环境缺少 `pytest`
- 因此仓库测试文件已经补齐，但本地验证主要通过 `conda run -n appo python -c ...` 和 CLI smoke 完成

## 推荐验证命令

在具备完整依赖的 `appo` 环境中执行：

```bash
conda run -n appo python -m pytest tests/test_resource_graph_granularity.py -q
conda run -n appo python -m pytest tests/v2/test_gnn_encoder_v2.py tests/v2/test_agent_v2.py tests/v2/test_smoke.py -q
conda run -n appo python experiments/train_v2.py --epochs 2 --device cpu --log_interval 1 --save_interval 999999 --mini_batch_size 32
conda run -n appo python experiments/train.py --algo mappo --epochs 2 --device cpu --log_interval 1 --save_interval 999999 --mini_batch_size 32
```

## 验收标准

- `res_x.shape == (M + K + 2, 2)`
- `res_edge_index.shape[1] == R * (R - 1)`
- `obs.shape[0] == cfg.obs_dim`
- 资源负载更新命中具体 `uav_m` / `sat_k` / `cloud`
- v1 与 v2 的 episode smoke 均无 shape error / index error
- 2-epoch CLI smoke 可跑通并输出训练日志
