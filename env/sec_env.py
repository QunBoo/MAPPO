"""
Satellite Edge Computing (SEC) Multi-Agent Environment — AMAPPO project.

4-tier architecture:
    IoTD (N=100) -> UAV (M=4, agents) -> LEO satellites (K=8) -> Cloud (1)

Each UAV agent manages a group of IoTDs, controls task offloading decisions,
bandwidth/compute allocation, and its own 2-D movement.

Observation per agent : dynamic dims
    [task_features(5) | upstream(20) | servers(R=M+K+2) | prev_action(8)]
Action per agent      :  8 dims  [offload_logits(4) | bandwidth(1) | compute(1) | displacement(2)]
"""

from __future__ import annotations

import math
import collections
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import torch

from env.dag_generator import DAGGenerator
from env.channel_model import (
    ChannelModel,
    P_IOTD_MAX, P_UAV, P_SAT,
    BW_G2U, BW_SAT, BW_ISL,
)

# ---------------------------------------------------------------------------
# Physical / model constants
# ---------------------------------------------------------------------------
_KAPPA_D: float = 5e-27   # IoTD effective switched capacitance
_KAPPA_U: float = 1e-28   # UAV
_KAPPA_L: float = 1e-28   # LEO satellite
_KAPPA_C: float = 1e-28   # Cloud

_F_LOCAL: float = 0.8e9   # Hz
_F_UAV:   float = 3.0e9
_F_SAT:   float = 4.5e9   # default; randomised per task in [4, 5] GHz
_F_CLOUD: float = 10.0e9

# LEO orbit altitude (m) — used to place satellites above scene
_SAT_ALT: float = 550e3   # 550 km

# Offload destination indices
_OFF_LOCAL = 0
_OFF_UAV   = 1
_OFF_SAT   = 2
_OFF_CLOUD = 3

# Fixed action dimensions
ACTION_DIM = 8    # 4 + 1 + 1 + 2

# Max predecessor slots (padded)
_MAX_PRED_SLOTS = 5
_TASK_FEATURE_DIM = 5
_UPSTREAM_FEATURE_DIM = _MAX_PRED_SLOTS * 4
_BASE_OBS_DIM = _TASK_FEATURE_DIM + _UPSTREAM_FEATURE_DIM + ACTION_DIM


class SECEnv:
    """
    Gym-style multi-agent environment for satellite edge computing.

    Parameters
    ----------
    config : object
        Namespace / dataclass with attributes:
          N, M, K, J         — counts (IoTDs, UAVs, satellites, tasks-per-DAG)
          area_size           — square side length [m]
          dt                 — timestep duration [s]
          max_steps          — episode horizon
          eta_t, eta_e       — reward coefficients (default 0.5 each)
          lambda_c           — constraint violation penalty (default 10.0)
          v_max              — UAV max speed [m/s] (default 30.0)
          d_min              — minimum UAV-to-UAV distance [m] (default 3.0)
          H_uav              — UAV flight altitude [m] (default 50.0)
    """

    def __init__(self, config) -> None:
        if hasattr(config, "sync_derived_fields"):
            config.sync_derived_fields()
        self.N         = int(config.N)
        self.M         = int(config.M)    # number of agents == number of UAVs
        self.K         = int(config.K)
        self.J         = int(config.J)
        self.area      = float(config.area_size)
        self.dt        = float(config.dt)
        self.max_steps = int(config.max_steps)

        self.eta_t   = float(getattr(config, "eta_t",    0.5))
        self.eta_e   = float(getattr(config, "eta_e",    0.5))
        self.lam     = float(getattr(config, "lambda_c", 10.0))
        self.v_max   = float(getattr(config, "v_max",    30.0))
        self.d_min   = float(getattr(config, "d_min",    3.0))
        self.H_uav   = float(getattr(config, "H_uav",    50.0))
        self.resource_node_count = int(getattr(config, "resource_node_count", self.M + self.K + 2))

        self.n_agents  = self.M
        self.obs_dim   = _BASE_OBS_DIM + self.resource_node_count
        self.action_dim = ACTION_DIM

        self._dag_gen = DAGGenerator()
        self._channel = ChannelModel()

        # Placeholders — populated by reset()
        self.uav_pos: np.ndarray = np.zeros((self.M, 3))   # (M, 3)  x,y,z
        self.sat_pos: np.ndarray = np.zeros((self.K, 3))   # (K, 3)
        self.iotd_pos: np.ndarray = np.zeros((self.N, 3))  # (N, 3)

        self.dags: List[nx.DiGraph] = []
        self.topo_orders: List[List[int]] = []
        self.task_pointers: List[int] = []   # index into topo_orders[m]
        self.task_done: List[np.ndarray] = []  # bool array per DAG
        self.node_to_idx: List[dict] = []    # stable node -> array index map

        self.prev_actions: np.ndarray = np.zeros((self.M, ACTION_DIM))
        self._step_count: int = 0
        self._done: bool = False

        # Per-resource EMA load tracking [local | UAV nodes | SAT nodes | cloud]
        self.resource_loads: np.ndarray = np.zeros(self.resource_node_count, dtype=np.float32)

        # Per-agent task completion times (for deadline constraint)
        self._T_max: float = 5.0 * self.dt   # simple default deadline per task

        # Cloud is at a fixed logical position (use a satellite as relay)
        # For rate computation we treat cloud link as: UAV -> nearest sat -> cloud
        # represented as two hops. Cloud compute is separate.

    # ==========================================================================
    # reset
    # ==========================================================================

    def reset(self) -> Dict[int, np.ndarray]:
        """
        Reset environment: generate new DAGs, randomise positions.

        Returns
        -------
        dict
            {agent_id: obs_array (obs_dim,)} for each agent in [0, M).
        """
        rng = np.random.default_rng()

        # --- UAV positions: random in [0.1*area, 0.9*area] x [0.1*area, 0.9*area] x H_uav ---
        xy = rng.uniform(0.1 * self.area, 0.9 * self.area, size=(self.M, 2))
        self.uav_pos = np.column_stack([xy, np.full(self.M, self.H_uav)])

        # --- IoTD positions: random on ground within area ---
        xy_iotd = rng.uniform(0.0, self.area, size=(self.N, 2))
        self.iotd_pos = np.column_stack([xy_iotd, np.zeros(self.N)])

        # --- Satellite positions: uniformly spaced at altitude ---
        # Spread K satellites over the 1000x1000 area at altitude _SAT_ALT
        angle_step = 2 * math.pi / self.K
        sat_xy = np.array(
            [
                [
                    self.area / 2 + (self.area / 3) * math.cos(i * angle_step),
                    self.area / 2 + (self.area / 3) * math.sin(i * angle_step),
                ]
                for i in range(self.K)
            ]
        )
        self.sat_pos = np.column_stack([sat_xy, np.full(self.K, _SAT_ALT)])

        # --- Generate one DAG per UAV group ---
        self.dags = []
        self.topo_orders = []
        self.task_pointers = []
        self.task_done = []
        self.node_to_idx = []

        for m in range(self.M):
            dag = self._dag_gen.generate(num_tasks=self.J)
            topo = self._dag_gen.topological_sort(dag)
            self.dags.append(dag)
            self.topo_orders.append(topo)
            # Skip virtual source (node 0) at start
            self.task_pointers.append(0)
            # Build a stable node -> index mapping using sorted order
            nodes_sorted = sorted(dag.nodes())
            self.node_to_idx.append({n: i for i, n in enumerate(nodes_sorted)})
            self.task_done.append(np.zeros(len(nodes_sorted), dtype=bool))

        self.prev_actions = np.zeros((self.M, ACTION_DIM))
        self._step_count  = 0
        self._done        = False
        self.resource_loads = np.zeros(self.resource_node_count, dtype=np.float32)

        return self._get_obs_dict()

    # ==========================================================================
    # step
    # ==========================================================================

    def step(
        self, action_dict: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, dict]:
        """
        Execute one global timestep.

        Parameters
        ----------
        action_dict : {agent_id: action_array (8,)}

        Returns
        -------
        obs_dict, rew_dict, done, info
        """
        assert not self._done, "Environment is done. Call reset() first."

        self._step_count += 1

        # --- Parse and apply actions ---
        parsed = {}   # agent_id -> (offload_idx, bw_alloc, comp_alloc, delta_xy)
        for m in range(self.M):
            action = action_dict.get(m, np.zeros(8))
            action = np.asarray(action, dtype=np.float32)

            logits   = action[0:4]
            bw       = float(np.clip(self._sigmoid(action[4]), 0.01, 1.0))
            comp     = float(np.clip(self._sigmoid(action[5]), 0.01, 1.0))
            # Raw displacement in metres; no tanh saturation so speed can exceed v_max
            delta_xy = np.asarray(action[6:8], dtype=np.float64) * 50.0  # scale: ±50 m/step

            off_idx = int(np.argmax(logits))
            parsed[m] = (off_idx, bw, comp, delta_xy)

        # --- Move UAVs ---
        old_uav_pos = self.uav_pos.copy()
        # Track out-of-area BEFORE clipping so Phi_3 can fire correctly
        self._out_of_area: List[bool] = []
        for m in range(self.M):
            _, _, _, delta_xy = parsed[m]
            # delta_xy is already in metres (raw, no v_max saturation)
            new_xy = self.uav_pos[m, :2] + delta_xy
            # Check out-of-area BEFORE clipping
            out_of_area = bool(new_xy[0] < 0.0 or new_xy[0] > self.area or
                               new_xy[1] < 0.0 or new_xy[1] > self.area)
            self._out_of_area.append(out_of_area)
            # THEN clip position to stay inside the area boundary
            self.uav_pos[m, 0] = np.clip(new_xy[0], 0.0, self.area)
            self.uav_pos[m, 1] = np.clip(new_xy[1], 0.0, self.area)
            # altitude is fixed

        # --- Compute rewards and check constraints ---
        rewards: Dict[int, float] = {}
        violation_counts = np.zeros(5, dtype=int)
        all_times:    List[float] = []
        all_energies: List[float] = []

        for m in range(self.M):
            off_idx, bw, comp, delta_xy = parsed[m]

            # Current task for this agent
            task_node = self._current_task(m)

            if task_node is None:
                # DAG finished — zero reward
                rewards[m] = 0.0
                continue

            dag   = self.dags[m]
            attrs = dag.nodes[task_node]
            D_in_bits  = attrs["D_in"]  * 8e6
            D_out_bits = attrs["D_out"] * 8e6
            C_cycles   = attrs["C"]  * 1e9  # Gcycles -> cycles

            # --- Transmission time ---
            T_trans = self._calc_T_trans(m, off_idx, bw, D_in_bits, D_out_bits)

            # --- Computation time ---
            T_comp, f_exec, kappa = self._calc_T_comp(m, off_idx, comp, C_cycles)

            T_i = T_trans + T_comp

            # --- Energy consumption ---
            P_tx = P_IOTD_MAX * bw   # scale tx power by bandwidth fraction
            E_tx = P_tx * T_trans
            E_comp = kappa * C_cycles * (f_exec ** 2)
            E_i = E_tx + E_comp

            all_times.append(T_i)
            all_energies.append(E_i)

            # --- Constraint violations ---
            phi = np.zeros(5, dtype=float)

            # Phi_1: task deadline
            if T_i > self._T_max:
                phi[0] = 1.0
                violation_counts[0] += 1

            # Phi_2: UAV collision
            for other in range(self.M):
                if other == m:
                    continue
                dist_xy = np.linalg.norm(self.uav_pos[m, :2] - self.uav_pos[other, :2])
                if dist_xy < self.d_min:
                    phi[1] = 1.0
                    violation_counts[1] += 1
                    break

            # Phi_3: UAV out of area (checked BEFORE clipping in move phase)
            if self._out_of_area[m]:
                phi[2] = 1.0
                violation_counts[2] += 1

            # Phi_4: UAV speed exceeded
            actual_disp = np.linalg.norm(self.uav_pos[m, :2] - old_uav_pos[m, :2])
            actual_speed = actual_disp / self.dt if self.dt > 0 else 0.0
            if actual_speed > self.v_max + 1e-6:
                phi[3] = 1.0
                violation_counts[3] += 1

            # Phi_5: Resource overload — penalise near-capacity compute allocation
            if comp > 0.95:
                phi[4] = 1.0
                violation_counts[4] += 1

            # Update the specific resource node selected by this offload route.
            resource_idx = self.resource_index_for_offload(m, off_idx)
            self.resource_loads[resource_idx] = (
                0.9 * self.resource_loads[resource_idx] + 0.1 * comp
            )

            # --- Reward ---
            _E_SCALE = 0.01  # scale E_i to the same magnitude as T_i (E_i ~100x larger otherwise)
            r_m = (
                -self.eta_t * T_i
                - self.eta_e * E_i * _E_SCALE
                - self.lam  * float(np.sum(phi))
            )
            rewards[m] = float(r_m)

        # --- Advance task queues ---
        for m in range(self.M):
            self._advance_task(m)

        # Store actions for next obs — store full ACTION_DIM (8) dims in prev_actions
        for m in range(self.M):
            raw_act = action_dict.get(m, np.zeros(ACTION_DIM))
            self.prev_actions[m] = np.asarray(raw_act, dtype=np.float32)[:ACTION_DIM]

        # --- Done check ---
        all_finished = all(self._is_dag_done(m) for m in range(self.M))
        self._done = all_finished or (self._step_count >= self.max_steps)

        # --- Info ---
        info = {
            "T_total":    float(np.mean(all_times))    if all_times    else 0.0,
            "E_total":    float(np.mean(all_energies)) if all_energies else 0.0,
            "violations": violation_counts.tolist(),
            "step":       self._step_count,
        }

        obs_dict = self._get_obs_dict()
        return obs_dict, rewards, self._done, info

    # ==========================================================================
    # Internal helpers
    # ==========================================================================

    # --- Observation ---

    def _get_obs_dict(self) -> Dict[int, np.ndarray]:
        return {m: self._build_obs(m) for m in range(self.M)}

    def _build_obs(self, m: int) -> np.ndarray:
        """Build observation for agent m with fine-grained resource states."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        ptr = 0

        dag  = self.dags[m]
        topo = self.topo_orders[m]

        task_node = self._current_task(m)

        # ---- task_features (5 dims) ----
        if task_node is not None:
            attrs = dag.nodes[task_node]
            d_in  = attrs["D_in"]
            d_out = attrs["D_out"]
            c_cyc = attrs["C"]
            # deadline remaining: rough estimate — steps left * dt
            steps_remaining = max(0, self.max_steps - self._step_count)
            deadline_rem = float(steps_remaining) * self.dt
            # task position in DAG: normalise by total nodes
            topo_pos = topo.index(task_node) / max(len(topo) - 1, 1)
            obs[0] = d_in
            obs[1] = d_out
            obs[2] = c_cyc
            obs[3] = deadline_rem
            obs[4] = topo_pos
        ptr = _TASK_FEATURE_DIM

        # ---- upstream_decisions (20 dims = 5 predecessors × 4 dims each) ----
        if task_node is not None:
            predecessors = list(dag.predecessors(task_node))
            for slot_idx, pred in enumerate(predecessors[:_MAX_PRED_SLOTS]):
                base = ptr + slot_idx * 4
                # Use prev_actions' offload decision for predecessors if available
                # (simplified: encode predecessor task attrs as proxy)
                p_attrs = dag.nodes[pred]
                obs[base + 0] = p_attrs["D_in"]
                obs[base + 1] = p_attrs["D_out"]
                obs[base + 2] = p_attrs["C"]
                obs[base + 3] = 1.0 if self._is_task_done(m, pred) else 0.0
        ptr += _UPSTREAM_FEATURE_DIM

        # ---- server_states (R dims) ----
        obs[ptr: ptr + self.resource_node_count] = self.resource_loads
        ptr += self.resource_node_count

        # ---- prev_action (8 dims) ----
        obs[ptr: ptr + ACTION_DIM] = self.prev_actions[m]

        return obs

    # --- Task queue management ---

    def _current_task(self, m: int) -> Optional[int]:
        """Return the current real task node for agent m, or None if done."""
        dag   = self.dags[m]
        topo  = self.topo_orders[m]
        vsrc  = dag.graph["virtual_source"]
        vsink = dag.graph["virtual_sink"]
        ptr   = self.task_pointers[m]

        while ptr < len(topo):
            node = topo[ptr]
            if node in (vsrc, vsink):
                ptr += 1
                continue
            if self._is_task_done(m, node):
                ptr += 1
                continue
            self.task_pointers[m] = ptr
            return node

        return None

    def _is_task_done(self, m: int, node: int) -> bool:
        """Check whether a specific node in DAG m is done."""
        idx = self.node_to_idx[m][node]
        return bool(self.task_done[m][idx])

    def _mark_task_done(self, m: int, node: int) -> None:
        idx = self.node_to_idx[m][node]
        self.task_done[m][idx] = True

    def _advance_task(self, m: int) -> None:
        """Mark the current task as done and advance the pointer."""
        node = self._current_task(m)
        if node is not None:
            self._mark_task_done(m, node)
            self.task_pointers[m] += 1

    def _is_dag_done(self, m: int) -> bool:
        """True if all real tasks in DAG m are completed."""
        dag  = self.dags[m]
        vsrc  = dag.graph["virtual_source"]
        vsink = dag.graph["virtual_sink"]
        for node in dag.nodes():
            if node in (vsrc, vsink):
                continue
            if not self._is_task_done(m, node):
                return False
        return True

    # --- Channel / timing helpers ---

    def _calc_T_trans(
        self, m: int, off_idx: int, bw: float,
        D_in_bits: float, D_out_bits: float
    ) -> float:
        """
        Compute transmission time for offloading task of agent m to off_idx.
        bw is the bandwidth fraction allocated (0..1).
        """
        uav_p = tuple(self.uav_pos[m])

        if off_idx == _OFF_LOCAL:
            # No transmission — data stays at device
            return 0.0

        if off_idx == _OFF_UAV:
            # IoTD -> UAV (G2U link)
            # Pick the IoTD closest to this UAV
            iotd_p = self._nearest_iotd(m)
            rate = ChannelModel.g2u_rate(
                uav_pos=uav_p,
                device_pos=tuple(iotd_p),
                tx_power_w=P_IOTD_MAX * bw,
            )
            rate = max(rate, 1.0)
            return D_in_bits / rate

        if off_idx == _OFF_SAT:
            # IoTD -> UAV -> Satellite
            # G2U hop
            iotd_p = self._nearest_iotd(m)
            rate_g2u = ChannelModel.g2u_rate(
                uav_pos=uav_p,
                device_pos=tuple(iotd_p),
                tx_power_w=P_IOTD_MAX * bw,
            )
            rate_g2u = max(rate_g2u, 1.0)

            # U2S hop
            sat_p = tuple(self.sat_pos[self._nearest_sat(m)])
            rate_u2s = ChannelModel.u2s_rate(
                sat_pos=sat_p,
                uav_pos=uav_p,
                tx_power_w=P_UAV,
            )
            rate_u2s = max(rate_u2s, 1.0)

            # Bottleneck: sum of hop times
            T_g2u = D_in_bits / rate_g2u
            T_u2s = D_in_bits / rate_u2s
            return T_g2u + T_u2s

        if off_idx == _OFF_CLOUD:
            # IoTD -> UAV -> Satellite -> Cloud
            iotd_p = self._nearest_iotd(m)
            rate_g2u = ChannelModel.g2u_rate(
                uav_pos=uav_p,
                device_pos=tuple(iotd_p),
                tx_power_w=P_IOTD_MAX * bw,
            )
            rate_g2u = max(rate_g2u, 1.0)

            sat_p = tuple(self.sat_pos[self._nearest_sat(m)])
            rate_u2s = ChannelModel.u2s_rate(
                sat_pos=sat_p,
                uav_pos=uav_p,
                tx_power_w=P_UAV,
            )
            rate_u2s = max(rate_u2s, 1.0)

            # S2C: distance between satellite and a notional cloud gateway (~600 km)
            rate_s2c = ChannelModel.s2c_rate(dist_km=600.0, tx_power_w=P_SAT)
            rate_s2c = max(rate_s2c, 1.0)

            T_g2u = D_in_bits / rate_g2u
            T_u2s = D_in_bits / rate_u2s
            T_s2c = D_in_bits / rate_s2c
            return T_g2u + T_u2s + T_s2c

        return 0.0

    def _calc_T_comp(
        self, m: int, off_idx: int, comp: float, C_cycles: float
    ) -> Tuple[float, float, float]:
        """
        Compute computation time T_comp, effective frequency f_exec, and
        energy capacitance coefficient kappa.

        Returns (T_comp, f_exec, kappa).
        """
        comp = max(comp, 0.01)

        if off_idx == _OFF_LOCAL:
            f = _F_LOCAL * comp
            kappa = _KAPPA_D
        elif off_idx == _OFF_UAV:
            f = _F_UAV * comp
            kappa = _KAPPA_U
        elif off_idx == _OFF_SAT:
            # Randomise satellite frequency in [4, 5] GHz
            f_sat = np.random.uniform(4e9, 5e9)
            f = f_sat * comp
            kappa = _KAPPA_L
        else:  # cloud
            f = _F_CLOUD * comp
            kappa = _KAPPA_C

        f = max(f, 1e6)   # floor to avoid division by zero
        T_comp = C_cycles / f
        return T_comp, f, kappa

    # --- Geometry utilities ---

    def _nearest_iotd(self, m: int) -> np.ndarray:
        """Return position of the IoTD closest to UAV m (in 2-D)."""
        uav_xy = self.uav_pos[m, :2]
        dists  = np.linalg.norm(self.iotd_pos[:, :2] - uav_xy, axis=1)
        return self.iotd_pos[int(np.argmin(dists))]

    def _nearest_sat(self, m: int) -> int:
        """Return index of the satellite closest to UAV m."""
        uav_xyz = self.uav_pos[m]
        dists   = np.linalg.norm(self.sat_pos - uav_xyz, axis=1)
        return int(np.argmin(dists))

    # --- Math utilities ---

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-float(np.clip(x, -20, 20))))

    # --- Resource graph helpers ---

    def local_resource_index(self) -> int:
        return 0

    def uav_resource_index(self, uav_id: int) -> int:
        return 1 + int(uav_id)

    def first_sat_resource_index(self) -> int:
        return 1 + self.M

    def sat_resource_index(self, sat_id: int) -> int:
        return self.first_sat_resource_index() + int(sat_id)

    def cloud_resource_index(self) -> int:
        return self.resource_node_count - 1

    def resource_index_for_offload(self, agent_id: int, offload_idx: int) -> int:
        if offload_idx == _OFF_LOCAL:
            return self.local_resource_index()
        if offload_idx == _OFF_UAV:
            return self.uav_resource_index(agent_id)
        if offload_idx == _OFF_SAT:
            return self.sat_resource_index(self._nearest_sat(agent_id))
        return self.cloud_resource_index()

    def _resource_capacities(self) -> np.ndarray:
        capacities = np.empty(self.resource_node_count, dtype=np.float32)
        capacities[self.local_resource_index()] = _F_LOCAL
        for m in range(self.M):
            capacities[self.uav_resource_index(m)] = _F_UAV
        for k in range(self.K):
            capacities[self.sat_resource_index(k)] = _F_SAT
        capacities[self.cloud_resource_index()] = _F_CLOUD
        return capacities

    def _resource_node_meta(self) -> List[dict]:
        node_meta = [{"name": "local", "type": "local", "index": self.local_resource_index()}]
        for m in range(self.M):
            node_meta.append({
                "name": f"uav_{m}",
                "type": "uav",
                "index": self.uav_resource_index(m),
                "uav_id": m,
            })
        for k in range(self.K):
            node_meta.append({
                "name": f"sat_{k}",
                "type": "sat",
                "index": self.sat_resource_index(k),
                "sat_id": k,
            })
        node_meta.append({
            "name": "cloud",
            "type": "cloud",
            "index": self.cloud_resource_index(),
        })
        return node_meta

    def get_resource_graph_data(self, include_meta: bool = False):
        capacities = self._resource_capacities()
        max_capacity = float(capacities.max()) if capacities.size else 1.0
        res_x = np.column_stack([
            self.resource_loads,
            capacities / max(max_capacity, 1.0),
        ]).astype(np.float32)

        res_edges = [
            [i, j]
            for i in range(self.resource_node_count)
            for j in range(self.resource_node_count)
            if i != j
        ]
        res_edge_index = torch.tensor(res_edges, dtype=torch.long).t().contiguous()
        res_x_t = torch.tensor(res_x, dtype=torch.float32)

        if include_meta:
            return res_x_t, res_edge_index, self._resource_node_meta()
        return res_x_t, res_edge_index

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def resource_graph(self) -> dict:
        """Fine-grained resource graph info keyed by stable node names."""
        capacities = self._resource_capacities()
        graph = {}
        for meta in self._resource_node_meta():
            idx = meta["index"]
            graph[meta["name"]] = {
                "load": float(self.resource_loads[idx]),
                "capacity": float(capacities[idx]),
                "type": meta["type"],
            }
        return graph

    @property
    def done(self) -> bool:
        return self._done

    def observation_space_dim(self) -> int:
        return self.obs_dim

    def action_space_dim(self) -> int:
        return self.action_dim
