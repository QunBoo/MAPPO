"""
DAG Generator for AMAPPO satellite edge computing project.

Generates random Directed Acyclic Graphs (DAGs) representing IoT task
applications. Each DAG models one IoTD's computation workload where nodes
are tasks and edges are data-flow dependencies.

Node attributes (for non-virtual nodes):
    D_in  : input data size  [0.8, 4.0] MB
    D_out : output data size [0.4, 1.0] MB
    C     : required CPU cycles [1, 3] Gcycles

Virtual source (node 0) and virtual sink (node num_tasks+1) carry all-zero
attributes and serve as topological anchors.
"""

from __future__ import annotations

import collections
from typing import List, Optional

import networkx as nx
import numpy as np


class DAGGenerator:
    """Generate random layered DAGs for task scheduling experiments."""

    # Node attribute ranges
    D_IN_RANGE = (0.8, 4.0)   # MB
    D_OUT_RANGE = (0.4, 1.0)  # MB
    C_RANGE = (1.0, 3.0)       # Gcycles

    # Layer count range (excluding virtual source/sink)
    L_RANGE = (3, 5)

    def generate(
        self,
        num_tasks: int = 20,
        num_layers: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> nx.DiGraph:
        """
        Generate a random layered DAG.

        Parameters
        ----------
        num_tasks : int
            Number of real (non-virtual) tasks. Default 20.
        num_layers : int or None
            Number of layers (1..L). If None, sampled uniformly from
            [L_RANGE[0], L_RANGE[1]].
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        nx.DiGraph
            A DAG where node 0 is the virtual source and node
            num_tasks+1 is the virtual sink.  All other nodes are real
            tasks with attributes D_in, D_out, C.
        """
        rng = np.random.default_rng(seed)

        # --- choose layer count ---
        if num_layers is None:
            num_layers = int(rng.integers(self.L_RANGE[0], self.L_RANGE[1] + 1))

        if num_layers is not None and not (1 <= num_layers <= num_tasks):
            raise ValueError(f"num_layers must be in [1, {num_tasks}], got {num_layers}")

        # --- assign real tasks (1 .. num_tasks) to layers ---
        # Each layer gets at least one task; remainder are distributed randomly.
        real_ids = list(range(1, num_tasks + 1))
        rng.shuffle(real_ids)

        # Seed each layer with one task, then randomly assign the rest
        layer_of: dict[int, int] = {}
        for layer_idx, task_id in enumerate(real_ids[:num_layers]):
            layer_of[task_id] = layer_idx + 1  # layers are 1-indexed

        remaining = real_ids[num_layers:]
        for task_id in remaining:
            layer_of[task_id] = int(rng.integers(1, num_layers + 1))

        # Build layer -> task-list mapping
        layers: dict[int, List[int]] = {l: [] for l in range(1, num_layers + 1)}
        for task_id, layer_idx in layer_of.items():
            layers[layer_idx].append(task_id)

        # --- build graph ---
        dag = nx.DiGraph()

        virtual_source = 0
        virtual_sink = num_tasks + 1

        # Add virtual nodes with zero attributes
        dag.add_node(virtual_source, D_in=0.0, D_out=0.0, C=0.0)
        dag.add_node(virtual_sink,   D_in=0.0, D_out=0.0, C=0.0)

        # Add real task nodes with random attributes
        for task_id in real_ids:
            dag.add_node(
                task_id,
                D_in=float(rng.uniform(*self.D_IN_RANGE)),
                D_out=float(rng.uniform(*self.D_OUT_RANGE)),
                C=float(rng.uniform(*self.C_RANGE)),
            )

        # Virtual source -> all layer-1 tasks
        for task_id in layers[1]:
            dag.add_edge(virtual_source, task_id)

        # Edges between consecutive layers (random subset, guaranteed >=1)
        for l in range(1, num_layers):
            src_layer = layers[l]
            dst_layer = layers[l + 1]
            # Each destination node gets at least one predecessor from src_layer
            used_edges: set[tuple[int, int]] = set()
            for dst in dst_layer:
                src = int(rng.choice(src_layer))
                used_edges.add((src, dst))
            # Each source node has at least one successor in dst_layer
            for src in src_layer:
                if not any(s == src for s, _ in used_edges):
                    dst = int(rng.choice(dst_layer))
                    used_edges.add((src, dst))
            # Optionally add a few extra random cross-edges
            extra = int(rng.integers(0, max(1, len(src_layer) * len(dst_layer) // 4) + 1))
            for _ in range(extra):
                src = int(rng.choice(src_layer))
                dst = int(rng.choice(dst_layer))
                used_edges.add((src, dst))
            dag.add_edges_from(used_edges)

        # All last-layer tasks -> virtual sink
        for task_id in layers[num_layers]:
            dag.add_edge(task_id, virtual_sink)

        # Sanity check
        assert nx.is_directed_acyclic_graph(dag), "Generated graph is NOT a DAG!"

        # Store virtual node ids as graph attributes for downstream use
        dag.graph["virtual_source"] = virtual_source
        dag.graph["virtual_sink"] = virtual_sink

        return dag

    # ------------------------------------------------------------------
    # Topological sort via Kahn's algorithm
    # ------------------------------------------------------------------

    def topological_sort(self, dag: nx.DiGraph) -> List[int]:
        """
        Return a topological ordering of DAG nodes using Kahn's algorithm.

        Parameters
        ----------
        dag : nx.DiGraph
            A directed acyclic graph.

        Returns
        -------
        List[int]
            Node indices in topological order.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        in_degree = {node: dag.in_degree(node) for node in dag.nodes()}
        queue = collections.deque(
            node for node, deg in in_degree.items() if deg == 0
        )
        order: List[int] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for successor in dag.successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(order) != dag.number_of_nodes():
            raise ValueError(
                "Cycle detected: topological sort failed. "
                f"Processed {len(order)} / {dag.number_of_nodes()} nodes."
            )

        return order

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        dag: nx.DiGraph,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Draw the DAG with matplotlib.

        Parameters
        ----------
        dag : nx.DiGraph
        save_path : str or None
            If given, save the figure to this path.
        show : bool
            If True, call plt.show() interactively.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[visualize] matplotlib not available — skipping.")
            return

        # Use a hierarchical (topological) layout via graphviz if available,
        # otherwise fall back to spring layout.
        try:
            pos = nx.nx_agraph.graphviz_layout(dag, prog="dot")
        except Exception:
            # Compute layer-based positions from topological sort
            topo = self.topological_sort(dag)
            topo_rank = {node: rank for rank, node in enumerate(topo)}
            # Group nodes by topo rank to avoid overlap
            rank_groups: dict = {}
            for node, rank in topo_rank.items():
                rank_groups.setdefault(rank, []).append(node)
            pos = {}
            for rank, nodes_at_rank in rank_groups.items():
                for i, node in enumerate(nodes_at_rank):
                    pos[node] = (rank, i - len(nodes_at_rank) / 2)

        labels = {n: str(n) for n in dag.nodes()}
        source = dag.graph.get("virtual_source", min(dag.nodes()))
        sink = dag.graph.get("virtual_sink", max(dag.nodes()))
        node_colors = []
        for n in dag.nodes():
            if n == source:
                node_colors.append("lightgreen")
            elif n == sink:
                node_colors.append("salmon")
            else:
                node_colors.append("skyblue")

        fig, ax = plt.subplots(figsize=(14, 6))
        nx.draw(
            dag,
            pos=pos,
            labels=labels,
            node_color=node_colors,
            node_size=600,
            font_size=8,
            arrows=True,
            arrowsize=12,
            ax=ax,
        )
        ax.set_title("Random DAG (AMAPPO task graph)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[visualize] Figure saved to {save_path}")
        if show:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Verification / __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    gen = DAGGenerator()
    num_tasks = 20

    print("=" * 60)
    print("Generating DAG with J=20 tasks ...")
    dag = gen.generate(num_tasks=num_tasks, seed=42)

    num_nodes = dag.number_of_nodes()
    num_edges = dag.number_of_edges()
    print(f"  Nodes : {num_nodes}  (20 real + 2 virtual)")
    print(f"  Edges : {num_edges}")

    # --- acyclicity check ---
    is_dag = nx.is_directed_acyclic_graph(dag)
    print(f"  nx.is_directed_acyclic_graph : {is_dag}")
    if not is_dag:
        print("ERROR: graph is NOT a DAG!", file=sys.stderr)
        sys.exit(1)

    # --- topological sort ---
    topo_order = gen.topological_sort(dag)
    print(f"\nTopological order ({len(topo_order)} nodes):")
    print(" ->".join(str(n) for n in topo_order))

    # --- node attribute sample ---
    print("\nNode attributes (sample — first 5 real tasks):")
    print(f"  {'Node':>6}  {'D_in (MB)':>10}  {'D_out (MB)':>10}  {'C (Gcycles)':>12}")
    print("  " + "-" * 44)
    sample_nodes = [n for n in dag.nodes() if n not in (0, num_tasks + 1)][:5]
    for node in sample_nodes:
        attr = dag.nodes[node]
        print(
            f"  {node:>6}  {attr['D_in']:>10.3f}  {attr['D_out']:>10.3f}  {attr['C']:>12.3f}"
        )

    print(f"\nVirtual source (node 0) attrs : {dag.nodes[0]}")
    print(f"Virtual sink  (node {num_tasks + 1}) attrs : {dag.nodes[num_tasks + 1]}")

    # --- visualization (save only, no GUI) ---
    import os
    import tempfile
    save_path = os.path.join(tempfile.gettempdir(), "dag_test.png")
    try:
        gen.visualize(dag, save_path=save_path, show=False)
    except Exception as exc:
        print(f"[visualize] Skipped due to error: {exc}")

    print("\nAll checks passed.")
