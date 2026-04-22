# 仿真数据可视化技术文档 — `experiments/plot_results.py`

## 1. 概述

`plot_results.py` 是 AMAPPO / AMAPPOv2 / MAPPO 多算法对比实验的结果可视化脚本。它从 TensorBoard 事件日志中读取标量数据（reward、cost），自动聚合多随机种子结果，生成收敛曲线与柱状对比图。

### 核心能力

| 能力 | 说明 |
|------|------|
| 多算法对比 | 支持 AMAPPO、AMAPPOv2、MAPPO 三种算法的同图对比 |
| 多种子聚合 | 自动发现并聚合同一算法下不同 seed 的运行结果，计算 mean ± std |
| 收敛曲线 | 生成 reward / cost 随训练轮次的收敛曲线（含标准差填充区域） |
| 柱状对比 | 生成最终性能（最后 100 episode 均值）的柱状对比图 |
| 灵活选择 | 支持通过命令行参数选择要可视化的算法子集、自定义日志目录 |

---

## 2. 模块架构

```
plot_results.py
│
├── 数据读取层
│   ├── read_tb_scalars()    — 从单个 TB 日志目录读取指定 tag 的标量数据
│   ├── collect_runs()       — 从多种子目录收集同一 tag 的所有运行数据
│   └── align_and_stack()    — 对齐截断并堆叠为 (n_seeds, T) 矩阵
│
├── 可视化层
│   ├── plot_convergence()   — 收敛曲线（line + fill_between）
│   └── plot_bar_comparison()— 柱状对比图
│
├── 配置 / 注册
│   └── ALGO_STYLE           — 算法 → (显示名, 颜色) 的映射表
│
└── 入口
    ├── parse_args()         — 命令行参数解析
    └── main()               — 主流程编排
```

---

## 3. 数据读取层详解

### 3.1 `read_tb_scalars(log_dir, tag)`

**功能**：从单个 TensorBoard 事件目录中读取指定 tag 的标量时间序列。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `log_dir` | `str` | TensorBoard 事件文件所在目录 |
| `tag` | `str` | 标量标签名，如 `"episode/reward"` |

**返回**：`Tuple[np.ndarray, np.ndarray]` — `(steps, values)`，按 step 排序。

**容错机制**：
- 若 `tensorboard` 包未安装，打印提示并返回空数组
- 若指定 `tag` 不存在，打印可用 tag 列表并返回空数组

**数据流**：

```
EventAccumulator(log_dir) → Reload() → Tags()["scalars"]
  → 过滤 tag → Scalars(tag) → 提取 (step, value) → argsort 排序
```

### 3.2 `collect_runs(base_dir, tag)`

**功能**：收集同一算法下所有随机种子运行的标量值。

**目录结构假设**：

```
base_dir/               ← 多种子模式
├── seed_0/             ← 子目录，每个包含事件文件
├── seed_1/
└── seed_2/

base_dir/               ← 单种子模式（无子目录）
├── events.out.tfevents.*
```

**逻辑**：
1. 扫描 `base_dir` 下所有子目录并排序
2. 若无子目录，则将 `base_dir` 本身作为唯一数据源
3. 对每个目录调用 `read_tb_scalars()` 读取指定 tag
4. 过滤空结果，返回 `List[np.ndarray]`（各数组长度可能不同）

### 3.3 `align_and_stack(runs)`

**功能**：将不等长的多种子运行对齐为统一长度矩阵。

**策略**：截断至最短运行长度（`min_len`），然后 `np.stack` 为 `(n_seeds, T)` 形状。

> ⚠️ 若某一种子提前结束，其后续数据会被丢弃。这是保守策略，确保所有种子在相同时间步上对齐。

**边界情况**：若 `runs` 为空，返回 `None`。

---

## 4. 可视化层详解

### 4.1 算法样式注册表 — `ALGO_STYLE`

```python
ALGO_STYLE = {
    "amappo":    ("AMAPPO",   "tab:blue"),
    "amappo_v2": ("AMAPPOv2", "tab:green"),
    "mappo":     ("MAPPO",    "tab:orange"),
}
```

| 键 | 显示标签 | Matplotlib 颜色 |
|----|----------|-----------------|
| `amappo` | AMAPPO | 蓝色 |
| `amappo_v2` | AMAPPOv2 | 绿色 |
| `mappo` | MAPPO | 橙色 |

未注册的算法键会以键名本身作为标签，颜色为 `None`（由 matplotlib 自动分配）。

### 4.2 `plot_convergence(algo_data, output_path, title, ylabel)`

**功能**：绘制多种算法的收敛曲线（均值线 + 标准差填充区域）。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `algo_data` | `Dict[str, Optional[np.ndarray]]` | `{算法键: (n_seeds, T) 矩阵}`，值为 `None` 表示无数据 |
| `output_path` | `str` | 输出图片路径 |
| `title` | `str` | 图表标题 |
| `ylabel` | `str` | Y 轴标签 |

**绘制逻辑**：

```
对每个算法:
  mean = values.mean(axis=0)        # 跨种子取均值
  std  = values.std(axis=0)         # 跨种子取标准差
  xs   = np.arange(len(mean))       # X 轴为 episode 序号

  ax.plot(xs, mean, ...)            # 均值折线
  ax.fill_between(xs, mean-std, mean+std, alpha=0.25)  # 标准差阴影
```

**输出规格**：
- 图片尺寸：9 × 5 英寸
- DPI：150
- 网格：虚线，透明度 0.5
- 自动 `tight_layout()`
- 若无有效数据，关闭图形并跳过

### 4.3 `plot_bar_comparison(algo_last100, output_path, metric_name)`

**功能**：绘制算法最终性能柱状对比图。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `algo_last100` | `Dict[str, float]` | `{算法键: 最后100 episode均值}` |
| `output_path` | `str` | 输出图片路径 |
| `metric_name` | `str` | Y 轴指标名 |

**绘制逻辑**：
1. 过滤掉值为 0.0 的条目（表示无数据）
2. 为每个柱体添加数值标注（保留 3 位小数）
3. 柱体宽度自适应算法数量：`min(0.4, 1.6 / len(algos))`
4. 图表宽度自适应：`max(5, 2 * len(algos))`

---

## 5. 主流程详解 — `main()`

### 5.1 执行流程图

```
main()
  │
  ├─ 1. parse_args()                    解析命令行参数
  │
  ├─ 2. 确定选中的算法列表
  │     ├─ --algos 指定 → 校验合法性
  │     └─ 未指定 → 全部三种算法
  │
  ├─ 3. 解析各算法的日志目录
  │     ├─ --xxx_dir 显式指定 → 使用指定路径
  │     └─ 未指定 → {log_dir}/{algo_key}
  │
  ├─ 4. 收集 Reward 数据
  │     ├─ 检查目录是否存在
  │     ├─ collect_runs(base_dir, "episode/reward")
  │     └─ align_and_stack() → reward_stacked
  │
  ├─ 5. 生成标题后缀
  │     └─ 从有数据的算法提取标签，用 " vs " 连接
  │
  ├─ 6. 绘制 Reward 收敛曲线
  │     └─ plot_convergence() → convergence_reward.png
  │
  ├─ 7. 计算最后 100 episode 均值
  │     └─ stacked[:, -100:].mean()
  │
  ├─ 8. 绘制柱状对比图
  │     └─ plot_bar_comparison() → bar_comparison.png
  │
  ├─ 9. 收集 Cost 数据
  │     └─ collect_runs(base_dir, "episode/cost")
  │        → align_and_stack() → cost_stacked
  │
  └─ 10. 绘制 Cost 收敛曲线（仅有数据时）
        └─ plot_convergence() → convergence_cost.png
```

### 5.2 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--log_dir` | `str` | `runs` | TensorBoard 日志根目录 |
| `--algos` | `str` | `None` | 逗号分隔的算法列表，如 `amappo_v2,mappo` |
| `--amappo_dir` | `str` | `None` | 显式指定 AMAPPO 日志目录 |
| `--amappo_v2_dir` | `str` | `None` | 显式指定 AMAPPOv2 日志目录 |
| `--mappo_dir` | `str` | `None` | 显式指定 MAPPO 日志目录 |
| `--output_dir` | `str` | `figures` | 输出图片保存目录 |
| `--tag` | `str` | `episode/reward` | Reward 使用的 TB 标量标签 |

### 5.3 输出文件

| 文件 | 条件 | 说明 |
|------|------|------|
| `convergence_reward.png` | 始终生成（若有数据） | Reward 收敛曲线 |
| `bar_comparison.png` | 始终生成（若有数据） | 最终性能柱状对比 |
| `convergence_cost.png` | 仅有 cost 数据时生成 | Cost 收敛曲线 |

---

## 6. 典型使用场景

### 场景 1：默认全量对比

```bash
python experiments/plot_results.py --output_dir figures
```

自动从 `runs/` 下读取 `amappo`、`amappo_v2`、`mappo` 三个子目录的数据。

### 场景 2：指定算法子集

```bash
python experiments/plot_results.py --algos amappo_v2,mappo --output_dir figures
```

仅对比 AMAPPOv2 与 MAPPO。

### 场景 3：显式指定日志目录

```bash
python experiments/plot_results.py \
    --amappo_v2_dir runs/amappo_v2_lr3e-4 \
    --mappo_dir runs/mappo_baseline \
    --output_dir figures/ablation
```

### 场景 4：自定义 Reward Tag

```bash
python experiments/plot_results.py --tag "episode/normalized_reward"
```

---

## 7. 期望的 TensorBoard 日志目录结构

```
runs/
├── amappo/
│   ├── seed_0/
│   │   └── events.out.tfevents.xxxxx
│   ├── seed_1/
│   │   └── events.out.tfevents.xxxxx
│   └── seed_2/
│       └── events.out.tfevents.xxxxx
├── amappo_v2/
│   ├── seed_0/
│   │   └── events.out.tfevents.xxxxx
│   ├── seed_1/
│   │   └── events.out.tfevents.xxxxx
│   └── seed_2/
│       └── events.out.tfevents.xxxxx
└── mappo/
    ├── seed_0/
    │   └── events.out.tfevents.xxxxx
    ├── seed_1/
    │   └── events.out.tfevents.xxxxx
    └── seed_2/
        └── events.out.tfevents.xxxxx
```

**必须记录的 TB 标量标签**：

| Tag | 用途 |
|-----|------|
| `episode/reward` | Reward 收敛曲线 + 柱状图 |
| `episode/cost` | Cost 收敛曲线（可选） |

---

## 8. 关键设计决策

| 决策 | 理由 |
|------|------|
| 截断对齐（非插值） | 避免引入插值伪影，保证统计真实性 |
| 标准差填充（非置信区间） | 直观展示多种子散布程度 |
| 最后 100 episode 均值 | 反映训练收敛后的稳态性能，减少早期波动影响 |
| 容错式跳过 | 目录不存在或 tag 缺失时优雅跳过，不中断整体流程 |
| 懒加载 matplotlib | 仅在绘图时 import，减少无绘图需求时的启动开销 |

---

## 9. 依赖项

| 包 | 用途 | 是否必须 |
|----|------|----------|
| `numpy` | 数值计算 | 是 |
| `matplotlib` | 绑图 | 是 |
| `tensorboard` | 读取事件日志 | 是（缺失时打印提示） |

安装方式：

```bash
pip install numpy matplotlib tensorboard
```
