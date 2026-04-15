# 软件需求规格说明书：Asynchronous Deep Reinforcement Learning Approach算法复现项目

## 1. 引言

### 1.1 目的

本软件需求规格说明书旨在定义基于论文《Cost-aware Dependent Task Offloading and Resource Allocation for Satellite Edge Computing: An Asynchronous Deep Reinforcement Learning Approach》的算法复现项目需求。该项目将实现一个面向卫星边缘计算（SEC）环境的异步深度强化学习框架，用于解决依赖任务卸载与资源分配联合优化问题，最小化系统成本（延迟与能耗的加权和）。

### 1.2 范围

**实现的内容包括：**

| 模块 | 描述 |
|:---|:---|
| **系统建模模块** | 四层级SEC架构（IoTD-UAV-LEO卫星-云服务器）的数学建模，包括通信模型、计算模型、DAG任务模型 |
| **设备关联模块** | 基于一对一匹配理论的IoTD与UAV关联算法 |
| **任务调度模块** | 多应用任务序列（MATS）算法，实现多DAG合并与任务优先级排序 |
| **核心算法模块** | 异步图神经网络增强的多智能体近端策略优化（AMAPPO）算法 |
| **仿真环境模块** | 基于真实数据集（Alibaba cluster-trace-v2018）的实验环境构建 |
| **可视化与评估模块** | 收敛曲线、能耗/延迟对比图、消融实验结果展示 |

**不包括的内容：**
- 实际硬件部署（仅软件仿真）
- 真实卫星通信链路的物理层实现
- 其他对比算法（MADDPG、A-PPO、IPPO等）的完整重新实现（可使用现有开源实现）

---

## 2. 系统建模与数据需求

### 2.1 任务模型

**2.1.1 DAG应用模型**

| 参数 | 符号 | 数据类型 | 取值范围/说明 |
|:---|:---|:---|:---|
| IoTD集合 | $\mathcal{N} = \{1,2,...,N\}$ | 整数 | $N = 100$（默认） |
| UAV集合 | $\mathcal{M} = \{1,2,...,M\}$ | 整数 | $M = 4$（默认） |
| LEO卫星集合 | $\mathcal{K} = \{1,2,...,K\}$ | 整数 | $K = 8$（默认） |
| 每个DAG的任务数 | $\|V_n\|$ | 整数 | $J = 20$（默认），可变范围[10,50] |
| 任务输入数据大小 | $D_{n,j}^{in}$ | 浮点 | [0.8, 4.0] MB |
| 任务输出数据大小 | $D_{n,j}^{o}$ | 浮点 | [0.4, 1.0] MB |
| 所需CPU周期 | $C_{n,j}$ | 浮点 | [1, 3] Gcycles，可变至3 Gcycles |
| 应用延迟约束 | $T_n^{max}$ | 浮点 | [50, 60] s |

**2.1.2 通信链路模型**

| 链路类型 | 带宽参数 | 取值 | 信道模型 |
|:---|:---|:---|:---|
| 地面到UAV (G2U) | $B_{nm}^{G2U}$ | 20 MHz | Rician衰落，含LoS/NLoS分量 |
| UAV到地面 (U2G) | $B_{nm}^{U2G}$ | 20 MHz | Rician衰落 |
| 地面到卫星 (G2S) | $B^{G2S}$ | 15 MHz | Shadowed-Rician衰落（3种阴影等级） |
| 卫星到地面 (S2G) | $B^{S2G}$ | 15 MHz | Shadowed-Rician衰落 |
| UAV到卫星 (U2S) | $B^{U2S}$ | 15 MHz | Shadowed-Rician衰落 |
| 卫星到UAV (S2U) | $B^{S2U}$ | 15 MHz | Shadowed-Rician衰落 |
| 星间链路 (ISL) | $B_{kk'}$ | 1 GHz | 自由空间损耗模型 |
| 卫星到云 (S2C) | $B_{kc}$ | 1 GHz | 视距传输 |

**2.1.3 阴影等级参数（Rician信道）**

| 环境条件 | 阴影等级 | $g$ | $\Omega$ |
|:---|:---|:---|:---|
| 晴天 | Light | 19.4 | 1.29 |
| 雾天 | Average | 10.1 | 0.835 |
| 雷暴 | Heavy | 0.739 | $8.97 \times 10^{-4}$ |

**2.1.4 计算资源模型**

| 节点类型 | 计算能力 | 有效开关电容 | 发射功率 |
|:---|:---|:---|:---|
| IoTD本地 | $f_n^l = 0.8$ GHz | $\kappa_D = 5 \times 10^{-27}$ | $P_n \in [0, 1]$ W, $P_n^{max}=1$W |
| UAV | $f_m = 3$ GHz | $\kappa_U = 10^{-28}$ | $P_m = 2$ W |
| LEO卫星 | $f_k \in [4,5]$ GHz | $\kappa_L = 10^{-28}$ | $P_k = 5$ W |
| 云服务器 | $f_c = 10$ GHz | $\kappa_C = 10^{-28}$ | $P_c = 5$ W |

**2.1.5 UAV运动模型**

| 参数 | 符号 | 取值 | 约束 |
|:---|:---|:---|:---|
| 飞行高度 | $H_m$ | [40, 60] m | 固定高度或可变 |
| 最大速度 | $ve_m^{max}$ | 30 m/s | 速度约束 |
| 最小安全距离 | $d_{min}$ | 3 m | 防碰撞约束 |
| 飞行区域 | $Q_x^{max}, Q_y^{max}$ | 1 km × 1 km | 矩形区域约束 |
| 悬停功耗 | $p_m^{fly}$ | 速度函数 | 基于文献[28]的推进功率模型 |

### 2.2 状态、动作与奖励

**2.2.1 状态空间（Observation Space）**

对于智能体 $i$（UAV $m$ 或可见LEO卫星 $k$）在时间步 $t$：

$$o_i^t = \{\mathcal{L}^{(us)}(v_{i,t}), h_{v_{i,t}}, a_i^{t-1}, \{h_u^t\}_{u \in U_i}, T_{n,k}^{LEO}\}$$

| 状态组件 | 维度/类型 | 说明 |
|:---|:---|:---|
| 上游任务决策集 $\mathcal{L}^{(us)}(v_{i,t})$ | 可变长度列表 | 当前任务所有前驱任务的执行决策 |
| 任务嵌入 $h_{v_{i,t}}$ | 向量（经GNN编码） | 任务 $v_{i,t}$ 的图神经网络嵌入表示 |
| 上一动作 $a_i^{t-1}$ | 离散/连续混合 | 前一个任务的卸载决策和资源分配 |
| 服务器节点嵌入 $\{h_u^t\}$ | 向量列表 | 可用服务器（本地/UAV/卫星/云）的状态嵌入 |
| 卫星覆盖时间 $T_{n,k}^{LEO}$ | 标量 | LEO卫星 $k$ 对IoTD $n$ 的可见时间窗 |

**全局状态** $GS_i^t$：所有智能体局部观测的拼接。

**2.2.2 动作空间（Action Space）**

对于LEO卫星智能体 $k$：
$$a_k^t = \{z_{n,k}^t, P_n^t, x_{k,t}, f_{k,t}^k\}_{n \in N_k}$$

对于UAV智能体 $m$：
$$a_m^t = \{z_{n,m}^t, P_n^t, x_{m,t}, f_{m,t}^m, f_{m,t}^k, q_m^t\}_{n \in N_m}$$

| 动作组件 | 类型 | 范围 | 说明 |
|:---|:---|:---|:---|
| 带宽分配比 $z_{n,k}^t / z_{n,m}^t$ | 连续 | [0, 1] | 上行/下行带宽分配比例 |
| 发射功率 $P_n^t$ | 连续 | $[0, P_n^{max}]$ | IoTD发射功率控制 |
| 卸载决策 $x_{k,t} / x_{m,t}$ | 离散（one-hot） | {0,1}^4 | 选择本地/UAV/卫星/云执行 |
| UAV计算资源 $f_{m,t}^m$ | 连续 | $[0, F_m]$ | 分配给任务的UAV CPU频率 |
| 卫星计算资源 $f_{m,t}^k / f_{k,t}^k$ | 连续 | $[0, F_k]$ | 分配给任务的卫星CPU频率 |
| UAV位置 $q_m^t$ | 连续（2D坐标） | 矩形区域内 | UAV下一时刻的二维位置 |

**2.2.3 奖励函数**

$$r_i^t(o_i^t, a_i^t) = -\eta_t T_{i,t} - \eta_e E_{i,t} - \sum_{\iota=1}^{5} \lambda_\iota \cdot \Phi_\iota$$

| 惩罚项 $\Phi_\iota$ | 计算方式 | 说明 |
|:---|:---|:---|
| $\Phi_1$ | $\max(0, T_n^{comp} - T_n^{max})$ | 延迟约束违反 |
| $\Phi_2$ | $\max(0, \sum_{n,j} f_{n,j}^m - F_m)$ | UAV计算资源过载 |
| $\Phi_3$ | $\max(0, \sum_{n,j} f_{n,j}^k - F_k)$ | 卫星计算资源过载 |
| $\Phi_4$ | $\max(0, ve_m - ve_m^{max})$ | UAV速度约束违反 |
| $\Phi_5$ | $\max(0, d_{min} - \|q_m(t) - q_{m'}(t)\|)$ | UAV碰撞约束违反 |

权重参数：$\eta_t, \eta_e$ 为延迟和能耗权重；$\lambda_\iota$ 为各约束的惩罚系数。

---

## 3. 算法功能需求

### 3.1 第一阶段：设备关联（One-to-Many Matching）

| 功能ID | 功能描述 | 输入 | 输出 |
|:---|:---|:---|:---|
| F1.1 | 偏好配置计算 | IoTD位置、UAV位置、信道增益 | IoTD偏好列表（按上行速率降序）、UAV偏好列表（按能耗升序） |
| F1.2 | 初始化随机匹配 | IoTD集合$\mathcal{N}$、UAV集合$\mathcal{M}$ | 初始匹配方案 $y$ |
| F1.3 | 交换匹配检测 | 当前匹配 $y$、候选交换对 $(n, n')$ | 是否为交换阻塞对的布尔判断 |
| F1.4 | 双向交换稳定匹配 | 初始匹配 | 稳定匹配 $y^*$、关联矩阵 $B$、各UAV服务的IoTD数 $N_m$ |

**算法流程：** 实现Algorithm 1，复杂度 $O(N^2)$。

#### 设备关联（One-to-Many Matching）算法伪代码

Algorithm 1: One-to-many Matching Algorithm for Device Association

Input : The set of IoTDs N, the set of UAVs M
1   Initialization: Select a random matching y and perform calculations
      by Eqs. (43) and (44);
2   while No swap matching ym′ exists do
3       Select IoTD n ∈ N, y(n) = m and IoTD n′ ∈ y(m′);
4       if IoTDs pair (n, n′) is a swap matching then
5           y ← ym′;
6           Compute Eqs. (43) and (44);
7       end
8   end
9   Acquire optimal matching y*;
10  Calculate the device association strategy via Eq. (42);

Output: The number of IoTDs Nm for each UAV m and IoTD
        association vector B.

该算法通过迭代执行有效的交换匹配，直到达到双边交换稳定状态，即不存在能进一步增加至少一方效用且不降低任何相关方效用的交换对为止。
其中使用的公式如下：

##### 公式 (42) — 匹配指示函数
将匹配函数 $y$ 转化为设备匹配指示变量 $b_{n,m}$：
\[
b_{n,m} = 
\begin{cases} 
1, & \text{if } m = y(n); \\
0, & \text{otherwise}.
\end{cases}
\tag{42}
\]

##### 公式 (43) 和 (44) — 偏好效用函数
- 物联网设备 $n$ 的偏好基于上行数据速率：
\[
\Psi_n(y) = R_{nm}^{up}(y)
\tag{43}
\]
- 无人机 $m$ 的偏好基于最小化能耗（表示为负能耗）：
\[
\Psi_m(y) = -E_{n,j}^{d,n,m}(y)
\tag{44}
\]

其中：
- $R_{nm}^{up}$ 为设备 $n$ 到无人机 $m$ 的上行传输速率（定义见式(9)）；
- $E_{n,j}^{d,n,m}$ 为任务结果从无人机 $m$ 下行传输至设备 $n$ 的能耗（定义见式(21)）。

### 3.2 第二阶段：多应用任务序列（MATS）

| 功能ID | 功能描述 | 输入 | 输出 |
|:---|:---|:---|:---|
| F2.1 | DAG合并 | 多个应用DAG $\{G_r\}$、智能体可用服务器 $U_i$ | 合并后的DAG $G_i$（含虚拟入口/出口节点） |
| F2.2 | 期望相对剩余工作量(ERR)计算 | 合并DAG $G_i$、服务器集合 $U_i$ | 每个任务-服务器对的ERR矩阵 |
| F2.3 | 任务优先级排序 | ERR矩阵 | 任务排序 $rank(G_i)$（降序） |
| F2.4 | 优先级冲突解决 | 初始排序 | 调整后的ERR值和最终排序 |

**核心计算：** 
$$ERR[v_{i,j}, u] = \max_{v_{i,j'} \in Suc(v_{i,j})} \min_{u \in U_i} \{ERR[v_{i,j'}, u] + T_{i,j',u}^{Exe} + T_{i,j',j,u}^{Trans}\}$$

$$rank(v_{i,j}) = \sum_{u=1}^{|U|} \frac{ERR[v_{i,j}, u]}{|U|}$$

#### 设备关联（One-to-Many Matching）算法伪代码
##### 算法伪代码

```text
Algorithm 2: Multi-application Task Sequence Algorithm for Task Scheduling

Input : Set of DAG applications G = {G_n | n ∈ N_i} for agent i,
        agent index i
Output: Ordered task execution sequence for agent i

1  Merge all applications from IoTDs associated with agent i into a
   single DAG G_i = (V_i, E_i) by adding a dummy entry task v_i,entry
   and a dummy exit task v_i,exit with zero computation and
   communication costs;
2  for each task v_i,j ∈ V_i do
3      Compute the upward rank value rank(v_i,j) using Eq. (45);
4  end
5  Sort all tasks in V_i in non-increasing order of their rank(v_i,j)
   values;
6  return The sorted task sequence as the execution order for agent i;
```

##### 使用的公式

**公式 (45) — 任务向上排序值 (Upward Rank)**

任务 $v_{i,j}$ 的向上排序值定义为该任务平均计算开销与其所有直接后继任务中最大（向上排序值 + 平均通信开销）之和：

\[
rank(v_{i,j}) = \overline{w_{i,j}} + \max_{v_{i,k} \in succ(v_{i,j})} \left( \overline{c_{i,j,k}} + rank(v_{i,k}) \right)
\tag{45}
\]

其中：
- $\overline{w_{i,j}}$ 为任务 $v_{i,j}$ 在所有可用计算节点上的平均执行时间；
- $succ(v_{i,j})$ 表示任务 $v_{i,j}$ 的直接后继任务集合；
- $\overline{c_{i,j,k}}$ 表示任务 $v_{i,j}$ 与 $v_{i,k}$ 之间的平均数据传输时间。

### 3.3 第三阶段：AMAPPO训练与推理

#### 3.3.1 图感知编码器（GNN-based Encoder）

| 功能ID | 功能描述 | 技术细节 |
|:---|:---|:---|
| F3.1 | 任务DAG编码 | 使用GraphSAGE变体，分离上游/下游邻居聚合，ReLU激活 |
| F3.2 | 网络资源图编码 | 无向图GNN编码，聚合邻居服务器信息 |
| F3.3 | 全连接层与Max-pooling | 生成固定维度的图嵌入向量 |
| F3.4 | 嵌入拼接 | 任务图嵌入 + 网络图嵌入 → 解码器输入 |

**GNN架构细节：**
- 初始任务特征：$h_{v_{i,j}}^0 = \{H_{i,j}, C_{i,j}, |Pre_{i,j}|, |Suc_{i,j}|\}$
- 初始服务器特征：$h_u^0 = \{h_u, f_u^{avail}\}$
- 迭代步数 $p$：可配置（默认2-3层）

#### 3.3.2 图感知解码器（Decoder with Attention）

| 功能ID | 功能描述 | 技术细节 |
|:---|:---|:---|
| F3.5 | GRU状态更新 | 接收当前任务嵌入、上游决策、上一动作、服务器嵌入 |
| F3.6 | 注意力机制 | 计算历史任务嵌入的注意力分数，生成上下文向量 |
| F3.7 | 策略网络输出 | 基于上下文向量输出动作概率分布 $\pi(a_i^t\|o_i^t)$ |
| F3.8 | 价值网络输出 | 估计状态价值 $V(o_i^t)$ |

#### 3.3.3 异步训练机制（核心创新）

| 功能ID | 功能描述 | 关键机制 |
|:---|:---|:---|
| F3.9 | 双时钟系统 | 全局系统时钟 $t' \in GT$（固定短时长）+ 智能体本地时钟 $t_i$（可变步长） |
| F3.10 | 独立经验缓存 | 每个智能体维护独立转移缓存 $\xi_i$ |
| F3.11 | 异步数据收集 | 智能体完成当前任务后立即决策，无需等待其他智能体 |
| F3.12 | 周期性全局同步 | 按全局时钟将各智能体缓存数据汇入全局经验池 $MB$ |
| F3.13 | 策略网络更新 | PPO-clip目标函数，带梯度裁剪 |
| F3.14 | 价值网络更新 | TD学习，仅通过决策智能体自身观测的核传播梯度 |

**损失函数：**
- Critic Loss: $L_i^C(\theta) = \mathbb{E}_t[r_i^t + \gamma V_i(o_i^{t+1}; \theta) - V_i(o_i^t; \theta)]^2$
- Actor Loss: $L_i^A(\phi) = \mathbb{E}_t[\min(pp^t(\phi), \text{clip}(pp^t(\phi), 1-\epsilon_{clip}, 1+\epsilon_{clip})) \cdot \hat{A}^t(a_i^t, o_i^t)]$

**优势函数（GAE）：**
$$\hat{A}^t(a_i^t, o_i^t) = \sum_{j=1}^{|V_i|-t+1} (\gamma\varphi)^j(r^{t+j} + \gamma V_i(o_m^{t+j+1}) - V_m(o_m^{t+j}))$$

#### AMAPPO算法伪代码：
##### 算法伪代码

```text
Algorithm 3: Training Process of AMAPPO

1   Set memory buffer size MB = {};
2   Initialize transition buffer for each agent ξ₁, ..., ξ_I;
3   Initialize parameters φ and θ for the actor and critic networks;
4   Initialize RNN states h₁,π⁰, ..., h_I,π⁰ for actor network;
5   Initialize RNN states h₁,V⁰, ..., h_I,V⁰ for critic network;
6   for each epoch = 1 → ep do
7       for each time slot t′ = 1, 2, ..., GT do
8           for each available agent i do
9               Acquire ξ̃_i = (s_i^{t′}, o_i^{t′}, r_i^{t′});
10              ξ_i^{t′−1} ← ξ_i^{t′−1} ∪ ξ̃_i;
11              ξ_i ← ξ_i ∪ ξ_i^{t′−1};
12              a_i^{t′}, h_i,π^{t′} = π_i(o_i^{t′}, h_i,π^{t′−1}; φ);
13              h_i,V^{t′} = V_i(o_i^{t′}, h_i,V^{t′−1}; θ);
14              Acquire ξ_i = (s_i^{t′}, o_i^{t′}, h_i,π^{t′}, h_i,V^{t′}, a_i^{t′});
15          end
16          Execute action a_i^{t′};
17      end
18      for all agents i do
19          MB ← MB ∪ ξ_i;
20      end
21      Calculate rewards-to-go R̂ = Σ_{i=1}^{|V_i|−t′+1} γ_i r_i^{t′+1} on MB;
22      Calculate advantage estimate Â^t by Eq. (57) on MB;
23      Update θ by minimizing the loss function in Eq. (55);
24      Update φ using PPO-clip with the objective function in Eq. (56);
25  end

Output: The trained collaborative policy network π_i and critic network V_i
        for all the agents.
```

##### 使用的公式

**公式 (55) — Critic 网络损失函数 (TD误差)**

\[
L_{i}^{C}(\theta) = \mathbb{E}_{t}\left[ r_{i}^{t} + \gamma V_{i}(o_{i}^{t+1};\theta) - V_{i}(o_{i}^{t};\theta) \right]^{2}
\tag{55}
\]

其中：
- $r_i^t$ 为智能体 $i$ 在时刻 $t$ 的即时奖励；
- $\gamma$ 为折扣因子；
- $V_i(\cdot;\theta)$ 为参数 $\theta$ 下的价值函数。

**公式 (56) — Actor 网络损失函数 (PPO-Clip目标)**

\[
L_i^A(\phi) = \mathbb{E}_t \left[ \min \left( pp^t(\phi), \text{clip}(pp^t(\phi), 1 - \epsilon_{clip}, 1 + \epsilon_{clip}) \right) \cdot \hat{A}^t(a_i^t, o_i^t) \right]
\tag{56}
\]

其中：
- $pp^t(\phi) = \frac{\pi_i(a_i^t|o_i^t;\phi)}{\pi_i(a_i^t|o_i^t;\phi_{old})}$ 为新旧策略概率比；
- $\epsilon_{clip}$ 为裁剪范围超参数；
- $\hat{A}^t(\cdot)$ 为广义优势估计值。

**公式 (57) — 广义优势估计 (GAE)**

\[
\hat{A}^{t}(a_{i}^{t},o_{i}^{t}) = \sum_{j=1}^{|V_i|-t+1} (\gamma \varphi)^{j} \left( r^{t+j} + \gamma V_{i}(o_{i}^{t+j+1}) - V_{i}(o_{i}^{t+j}) \right)
\tag{57}
\]

其中：
- $\varphi \in [0,1]$ 为平滑参数，平衡估计的准确性与稳定性；
- $r^{t+j}$ 为第 $t+j$ 步的奖励。

### 3.4 对比算法接口（用于评估）

| 算法 | 需实现/集成的功能 |
|:---|:---|
| MAPPO | 同步版本，共享全局时钟，所有智能体同时决策 |
| MADDPG | 集中式训练分布式执行，使用确定性策略梯度 |
| A-PPO | 注意力机制PPO，单智能体或集中式 |
| IPPO | 独立PPO，无通信，各智能体独立训练 |
| AMAPPO+Match | 仅设备关联，无MATS |
| AMAPPO+Match+MATS | 完整方案 |

---

## 4. 技术栈与环境约束

### 4.1 硬件环境

| 组件 | 最低配置 | 推荐配置（论文使用） |
|:---|:---|:---|
| CPU | Intel Xeon或同等性能 | Intel Xeon Platinum 8370C @ 2.80GHz |
| GPU | NVIDIA GTX 1080Ti | NVIDIA GeForce GTX 4090 |
| 内存 | 32 GB | 64 GB |
| 存储 | 100 GB SSD | 500 GB NVMe SSD |

### 4.2 软件环境

| 类别 | 技术/工具 | 版本 | 用途 |
|:---|:---|:---|:---|
| 编程语言 | Python | 3.8+ | 主开发语言 |
| 深度学习框架 | PyTorch | 2.0+ | 神经网络实现 |
| GNN库 | PyTorch Geometric (PyG) | 2.3+ | 图神经网络层 |
| RL框架 | RLlib / Stable-Baselines3 / 自研 | 最新 | MARL算法基础（可选） |
| 科学计算 | NumPy, SciPy | 最新 | 数值计算 |
| 数据处理 | Pandas | 最新 | 数据集处理 |
| 可视化 | Matplotlib, Seaborn | 最新 | 绘图与结果展示 |
| 卫星仿真 | SaVi (外部工具) | 2023版 | LEO星座可视化与覆盖分析 |

### 4.3 数据集

| 数据集 | 来源 | 内容 | 处理方式 |
|:---|:---|:---|:---|
| Alibaba cluster-trace-v2018 | GitHub开源 | 4000台机器8天运行日志，含DAG依赖 | 采样子任务数量模拟远程IoTD应用（人脸识别、物体识别、姿态识别、手势识别） |

### 4.4 关键依赖库版本约束

```python
# requirements.txt 核心依赖
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
tensorboard>=2.8.0  # 训练可视化
```

### 4.5 超参数配置

| 参数 | 符号 | 值 | 可调范围 |
|:---|:---|:---|:---|
| 折扣因子 | $\gamma$ | 0.99 | [0.95, 0.999] |
| GAE平滑参数 | $\varphi$ | 0.95 | [0.9, 0.99] |
| PPO裁剪参数 | $\epsilon_{clip}$ | 0.2 | [0.1, 0.3] |
| 学习率 | $\alpha$ | $5 \times 10^{-4}$ | [$10^{-5}$, $10^{-3}$] |
| 小批量大小 | - | 128 | {64, 128, 256} |
| GRU隐藏层维度 | - | 64 | {32, 64, 128} |
| 训练轮数 | $ep$ | 1500 | 根据收敛情况调整 |
| 全局时间槽数 | $GT$ | 与环境相关 | 每epoch动态 |

---

## 5. 算法执行流程

### 5.1 整体流程图（对应论文Fig. 4）

```
┌─────────────────────────────────────────────────────────────┐
│                     系统初始化                               │
│  • IoTDs生成DAG应用 G_n = (V_n, E_n)                        │
│  • UAVs和LEO卫星定位                                        │
│  • 网络资源初始化                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段1: 一对一匹配算法（设备关联）                              │
│  输入: IoTD集合N, UAV集合M                                   │
│  1. 计算偏好配置（上行速率、能耗）                             │
│  2. 执行交换匹配迭代 → 稳定匹配 y*                           │
│  输出: 关联矩阵B, 各UAV服务IoTD数N_m                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段2: 多应用任务序列算法（任务调度）                          │
│  输入: 智能体i的N_i个IoTD的应用                              │
│  1. 合并应用到单一DAG G_i（添加虚拟入口/出口）                 │
│  2. 计算ERR值和任务优先级rank(G_i)                          │
│  输出: 合并DAG G_i, 任务排序rank(G_i)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段3: AMAPPO算法（卸载与资源分配）                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ GNN编码器        │→│ 异步决策制定      │→│ 策略&价值网络 │ │
│  │ • 编码任务DAG   │  │ • 观察状态       │  │ 更新         │ │
│  │ • 编码网络G_net │  │ • GNN解码器      │  │              │ │
│  │ • 生成嵌入      │  │ • 生成动作       │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                         ↑__________________________________│
│                         └──────── 循环直到所有任务完成 ──────┘
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     系统输出                                 │
│  • 卸载决策X  • 资源分配F  • UAV轨迹θ,β                      │
│  • 带宽分配Z  • 功率分配P                                    │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 AMAPPO训练详细流程（Algorithm 3）

```python
# 伪代码实现框架

# 初始化
memory_buffer_MB = {}
transition_buffers = {i: [] for i in agents}  # ξ_i
actor_params_φ, critic_params_θ = initialize_networks()
rnn_states_actor = {i: zero_state for i in agents}
rnn_states_critic = {i: zero_state for i in agents}

for epoch in range(1, ep+1):
    for global_time_slot t_prime in range(1, GT+1):
        # 异步决策阶段
        for each available agent i:
            # 获取转移数据（如果缓存中有上一时刻数据）
            if transition_buffers[i]:
                xi_hat = (s_i^{t'}, o_i^{t'}, r_i^{t'})
                transition_buffers[i].append(xi_hat)
            
            # 策略推理
            a_i^{t'}, h_i,π^{t'} = actor_network(o_i^{t'}, h_i,π^{t'-1}; φ)
            h_i,V^{t'} = critic_network(o_i^{t'}, h_i,V^{t'-1}; θ)
            
            # 存储当前转移起点
            xi = (s_i^{t'}, o_i^{t'}, h_i,π^{t'}, h_i,V^{t'}, a_i^{t'})
            transition_buffers[i].append(xi)
        
        # 执行动作（环境步进）
        execute_actions({a_i^{t'} for all available i})
    
    # 全局同步与训练
    for all agents i:
        memory_buffer_MB.update(transition_buffers[i])
        transition_buffers[i].clear()  # 清空已提交缓存
    
    # 计算回报和优势
    rewards_to_go = calculate_returns(memory_buffer_MB, γ)
    advantages = calculate_gae(memory_buffer_MB, φ, γ)  # Eq. (57)
    
    # 网络更新
    update_critic(θ, MSE_loss, rewards_to_go)  # Eq. (55)
    update_actor(φ, PPO_clip_loss, advantages)  # Eq. (56)
```

### 5.3 双时钟机制实现细节

| 时钟类型 | 特性 | 实现方式 |
|:---|:---|:---|
| 全局系统时钟 $GT$ | 固定短时长（如100ms），所有智能体同步 | 主循环控制，定期触发 |
| 智能体本地时钟 $t_i$ | 可变步长，取决于任务执行时间 | 每个智能体独立维护，任务完成即触发决策 |

**异步优势：** 智能体无需等待其他智能体完成当前任务，立即基于最新观测做决策，避免同步等待造成的空闲时间。

---

## 6. 验证指标

### 6.1 主要性能指标（与论文Fig. 9-13对应）

| 指标类别 | 指标名称 | 符号 | 计算方式 | 优化目标 |
|:---|:---|:---|:---|:---|
| **延迟指标** | 平均DAG完成延迟 | $T^{total}$ | $\frac{1}{N}\sum_{n=1}^N T_n^{comp}$ | 最小化 |
| | 单DAG完成延迟 | $T_n^{comp}$ | $\max_{j \in v_{n,exit}} FT_{n,j}$ | 约束满足 |
| **能耗指标** | 系统总能耗 | $E^{total}$ | $\sum_{n=1}^N E_n + \sum_{m=1}^M \max_{n \in N_m} T_n^{comp} \cdot p_m^{fly}$ | 最小化 |
| | IoTD n的能耗 | $E_n$ | $E_n^{exe} + E_n^{tran}$ | 最小化 |
| **综合成本** | 加权系统成本 | $Cost$ | $\eta_t T^{total} + \eta_e E^{total}$ | 最小化 |

### 6.2 训练过程指标

| 指标 | 说明 | 可视化方式 |
|:---|:---|:---|
| 回合奖励（Episode Reward） | 每回合累积奖励 | 学习曲线（Fig. 7, Fig. 8） |
| 策略熵（Policy Entropy） | 策略探索程度 | 辅助监控 |
| 价值函数损失（Value Loss） | Critic网络训练误差 | TensorBoard曲线 |
| 策略损失（Policy Loss） | Actor网络训练误差 | TensorBoard曲线 |
| 约束违反率 | 各约束违反的频率和幅度 | 训练日志 |

### 6.3 消融实验与对比实验（验证各组件贡献）

| 实验设置 | 对比算法/配置 | 验证目的 |
|:---|:---|:---|
| 完整方案 | AMAPPO+Match+MATS | 基准最优性能 |
| 无设备关联优化 | AMAPPO（随机匹配） | 验证一对一匹配算法的必要性 |
| 无任务调度优化 | AMAPPO+Match（无MATS） | 验证MATS算法的必要性 |
| 同步对比 | MAPPO（同步MARL） | 验证异步机制的优势 |
| 其他MARL基线 | MADDPG, A-PPO, IPPO | 验证AMAPPO相对于其他SOTA的优越性 |

### 6.4 参数敏感性实验

| 实验变量 | 变化范围 | 对应论文图表 |
|:---|:---|:---|
| IoTD数量 | 100 → 200 | Fig. 9 |
| DAG任务数量 | 10 → 50 | Fig. 10 |
| 任务CPU周期 | 1 → 3 Gcycles | Fig. 11 |
| LEO卫星数量 | 8 → 12 | Fig. 12 |
| 信道阴影等级 | Light/Average/Heavy | Fig. 13, Table IV |
| 学习率 | 0.0001, 0.0005, 0.001 | Fig. 8(a) |
| 小批量大小 | 64, 128 | Fig. 8(b) |

### 6.5 统计显著性要求

| 要求 | 说明 |
|:---|:---|
| 重复次数 | 每个实验设置至少运行5次随机种子 |
| 置信区间 | 绘制均值 ± 标准差阴影区域 |
| 收敛判定 | 连续100回合奖励波动 < 5%视为收敛 |

### 6.6 预期实验结果（复现目标）

基于论文报告，完整方案AMAPPO+Match+MATS应达到：

| 对比场景 | 性能提升 |
|:---|:---|
| vs. MAPPO | 能耗降低10.3%，延迟降低10.9%（200 IoTDs） |
| vs. IPPO | 能耗降低16.3%（50任务DAG） |
| vs. MADDPG | 显著更低的能耗和延迟，更好的稳定性 |
| 重阴影环境 | 能耗降低12.1%，延迟降低13.9% vs. A-PPO |

---

## 附录：关键公式速查

| 公式编号 | 内容 | 位置 |
|:---|:---|:---|
| (1) | UAV-IoTD距离 | Section III-A |
| (2)-(5) | UAV运动约束 | Section III-A |
| (6)-(7) | 卫星覆盖时间 | Section III-A |
| (8)-(17) | 各链路速率 | Section III-B |
| (18)-(40) | 各节点执行/传输时延与能耗 | Section III-C |
| (41) | 优化问题P1 | Section III-F |
| (42)-(44) | 匹配理论定义与效用 | Section IV-A |
| (45) | 任务优先级计算 | Section IV-B |
| (46)-(48) | GNN聚合规则 | Section IV-C |
| (49)-(52) | MDP状态/动作/奖励定义 | Section IV-C |
| (53)-(54) | 解码器注意力机制 | Section IV-C |
| (55)-(57) | AMAPPO损失函数与GAE | Section IV-C |

---

**文档版本：** 1.0  
**基于论文版本：** IEEE Transactions on Mobile Computing, DOI 10.1109/TMC.2025.3645456  
**编制日期：** 2026-04-15