#!/usr/bin/env python3
"""
FedLLM: NetworkX Distance Adjacency Matrix Inspector & Visualizer
=============================================================================
This script constructs a NetworkX graph from freeway sensor distance topologies
(such as PeMSD4 or PeMSD8) and extracts, analyzes, and visualizes its
Distance Adjacency Matrix using NetworkX's native linear algebra and graph APIs.

Features:
1. Spatial Graph Construction:
   - Builds nx.DiGraph (or nx.Graph) with physical edge weights ('distance').
   - Reports graph topological diagnostics (density, degree distributions, connectivity).

2. Distance Adjacency Representations:
   - Direct Edge Distance Matrix: nx.to_numpy_array(G, weight='distance', nonedge=...)
   - Thresholded Gaussian Kernel Matrix: A_ij = exp(-dist(i,j)^2 / sigma^2)
   - All-Pairs Shortest Path Distance Matrix: nx.floyd_warshall_numpy(G, weight='distance')
   - Pandas Adjacency DataFrame: nx.to_pandas_adjacency(G, weight='distance')
   - SciPy Sparse CSR Matrix: nx.adjacency_matrix(G, weight='distance')

3. Display & Export Options:
   - Terminal Tabular Print: Clean formatted display of submatrices or custom node subsets.
   - Publication-Quality Plot: Multi-panel heatmaps and distance distribution histograms.
   - CSV Export: Export full N x N matrix to disk.

Usage:
   python show_distance_matrix.py
   python show_distance_matrix.py --dataset PEMS04 --submatrix 10
   python show_distance_matrix.py --dataset PEMS08 --nodes 0,1,2,3,4,5
   python show_distance_matrix.py --mode gaussian --save_fig pems04_gaussian.png
   python show_distance_matrix.py --save_csv pems04_distance_matrix.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
# Use non-interactive Agg backend if no display is available
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ── Terminal ANSI Colors ──────────────────────────────────────────────────────
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
RESET   = "\033[0m"


def _c(text: str, colour: str) -> str:
    """Wrap text in ANSI color escapes when stdout supports it."""
    if sys.stdout.isatty() or os.environ.get("FORCE_COLOR"):
        return f"{colour}{text}{RESET}"
    return text


# =============================================================================
# 1. Graph Construction
# =============================================================================

def build_networkx_graph(
    distance_file: str,
    directed: bool = True,
) -> nx.Graph:
    """
    Constructs a NetworkX graph from a CSV distance file.

    Expected CSV columns: 'from', 'to', 'cost' (or 'distance'/'weight').
    Each edge receives both 'distance' and 'weight' attributes.
    """
    if not os.path.exists(distance_file):
        raise FileNotFoundError(f"Distance file not found: {distance_file}")

    dist_df = pd.read_csv(distance_file)

    # Standardize column headers
    col_map: Dict[str, str] = {}
    for col in dist_df.columns:
        clean = str(col).strip().lower()
        if clean in ["from", "src", "start", "0"]:
            col_map[col] = "from"
        elif clean in ["to", "dst", "end", "1"]:
            col_map[col] = "to"
        elif clean in ["cost", "distance", "dist", "weight", "2"]:
            col_map[col] = "cost"
    dist_df = dist_df.rename(columns=col_map)

    # Instantiate directed or undirected graph
    G = nx.DiGraph() if directed else nx.Graph()

    for _, row in dist_df.iterrows():
        u = int(row["from"])
        v = int(row["to"])
        dist = float(row["cost"])
        G.add_edge(u, v, distance=dist, weight=dist)

    return G


# =============================================================================
# 2. Matrix Extraction & Transformation
# =============================================================================

def get_distance_adjacency_matrix(
    G: nx.Graph,
    nodelist: Optional[List[int]] = None,
    nonedge_value: float = 0.0,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    Extracts the edge distance adjacency matrix using NetworkX's nx.to_numpy_array.

    Args:
        G: NetworkX Graph or DiGraph.
        nodelist: Ordered list of nodes. If None, sorted G.nodes() is used.
        nonedge_value: Value for pairs with no direct edge (e.g. 0.0, np.inf, or np.nan).
        zero_diagonal: If True, sets diagonal entries D(u, u) = 0.0.

    Returns:
        N x N NumPy array where entry (i, j) is the physical distance from node i to j.
    """
    if nodelist is None:
        nodelist = sorted(G.nodes())

    matrix = nx.to_numpy_array(
        G,
        nodelist=nodelist,
        weight="distance",
        nonedge=nonedge_value,
    )
    if zero_diagonal:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def get_gaussian_adjacency_matrix(
    dist_matrix_with_inf: np.ndarray,
    normalized_k: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """
    Computes the standard traffic GCN Thresholded Gaussian Kernel adjacency matrix:
        A_{ij} = exp(- dist(i, j)^2 / sigma^2)   if A_{ij} >= normalized_k, else 0
    Self-loops A_{ii} are set to 1.0.

    Args:
        dist_matrix_with_inf: N x N distance matrix where non-edges are np.inf.
        normalized_k: Threshold kappa in [0, 1].

    Returns:
        (A, sigma): Normalized adjacency matrix and the computed distance std-dev sigma.
    """
    finite_mask = ~np.isinf(dist_matrix_with_inf) & (dist_matrix_with_inf > 0)
    finite_distances = dist_matrix_with_inf[finite_mask]

    sigma = float(np.std(finite_distances)) if len(finite_distances) > 0 else 1.0
    if sigma == 0.0:
        sigma = 1.0

    gaussian_weights = np.zeros_like(dist_matrix_with_inf, dtype=np.float32)
    gaussian_weights[finite_mask] = np.exp(-np.square(dist_matrix_with_inf[finite_mask] / sigma))

    # Apply threshold kappa
    adj = np.where(gaussian_weights >= normalized_k, gaussian_weights, 0.0)
    np.fill_diagonal(adj, 1.0)
    return adj, sigma


def get_shortest_path_matrix(
    G: nx.Graph,
    nodelist: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Computes all-pairs shortest path distance matrix via NetworkX's Floyd-Warshall algorithm.
    Entry (i, j) is the minimum cumulative freeway path distance between i and j.
    """
    if nodelist is None:
        nodelist = sorted(G.nodes())
    return nx.floyd_warshall_numpy(G, nodelist=nodelist, weight="distance")


# =============================================================================
# 3. Terminal Formatting & Reporting
# =============================================================================

def print_graph_summary(G: nx.Graph, dataset_name: str, distance_file: str) -> None:
    """Prints comprehensive topological stats of the NetworkX graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    is_directed = G.is_directed()
    graph_type = "Directed (DiGraph)" if is_directed else "Undirected (Graph)"
    density = nx.density(G)

    print("=" * 80)
    print(_c(f"  NetworkX Sensor Graph Topology: {dataset_name}", BOLD + CYAN))
    print("=" * 80)
    print(f"  {_c('Source File:', BOLD)}       {distance_file}")
    print(f"  {_c('Graph Type:', BOLD)}        {graph_type}")
    print(f"  {_c('Total Stations (Nodes):', BOLD)} {num_nodes:,}")
    print(f"  {_c('Corridor Edges:', BOLD)}    {num_edges:,}")
    print(f"  {_c('Graph Density:', BOLD)}     {density:.4f} ({density * 100:.2f}% of all possible pairs)")

    if is_directed:
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        print(f"  {_c('In-Degree Range:', BOLD)}   min = {min(in_degrees)}, avg = {np.mean(in_degrees):.1f}, max = {max(in_degrees)}")
        print(f"  {_c('Out-Degree Range:', BOLD)}  min = {min(out_degrees)}, avg = {np.mean(out_degrees):.1f}, max = {max(out_degrees)}")
        scc = nx.number_strongly_connected_components(G)
        wcc = nx.number_weakly_connected_components(G)
        print(f"  {_c('Connectivity:', BOLD)}      {scc} Strongly Connected, {wcc} Weakly Connected Component(s)")
    else:
        degrees = [d for _, d in G.degree()]
        print(f"  {_c('Degree Range:', BOLD)}      min = {min(degrees)}, avg = {np.mean(degrees):.1f}, max = {max(degrees)}")
    print("-" * 80)


def print_matrix_table(
    matrix: np.ndarray,
    nodes: List[int],
    title: str,
    unit: str = "miles",
    max_display: int = 10,
    nonedge_repr: str = "  .  ",
) -> None:
    """Prints a formatted ASCII submatrix table for visual inspection."""
    display_nodes = nodes[:max_display]
    n = len(display_nodes)

    print()
    print(_c(f"--- {title} (Showing {n}x{n} Submatrix, Unit: {unit}) ---", BOLD + YELLOW))

    # Header row
    header = f"{'Node':>6} | " + " ".join(f"{node:>7}" for node in display_nodes)
    print(header)
    print("-" * len(header))

    # Rows
    for i, u in enumerate(display_nodes):
        row_str = f"{u:>6} | "
        for j, _ in enumerate(display_nodes):
            val = matrix[i, j]
            if np.isnan(val) or np.isinf(val) or (val == 0.0 and i != j):
                row_str += f"{nonedge_repr:>7} "
            elif i == j and val == 0.0:
                row_str += f"{'0.0':>7} "
            else:
                if val >= 100:
                    row_str += f"{val:>7.1f} "
                else:
                    row_str += f"{val:>7.2f} "
        print(row_str)

    if len(nodes) > max_display:
        print(f"... ({len(nodes) - max_display} additional nodes omitted in preview. Use --submatrix N to expand)")
    print()


def print_matrix_statistics(matrix: np.ndarray, name: str) -> None:
    """Prints linear algebra & sparsity statistics for the adjacency matrix."""
    total_elements = matrix.size
    finite_mask = ~np.isinf(matrix) & ~np.isnan(matrix) & (matrix > 0)
    num_edges = int(np.sum(finite_mask))
    sparsity = 100.0 * (1.0 - (num_edges / total_elements))

    print(_c(f"  Matrix Diagnostics [{name}]:", BOLD + GREEN))
    print(f"    - Dimensions:        {matrix.shape[0]} x {matrix.shape[1]} ({total_elements:,} total entries)")
    print(f"    - Non-Zero Edges:    {num_edges:,}")
    print(f"    - Matrix Sparsity:   {sparsity:.2f}% (sparse structure)")

    if num_edges > 0:
        edge_vals = matrix[finite_mask]
        print(f"    - Min Distance:      {np.min(edge_vals):.3f} miles")
        print(f"    - 25th Percentile:   {np.percentile(edge_vals, 25):.3f} miles")
        print(f"    - Median Distance:   {np.median(edge_vals):.3f} miles")
        print(f"    - 75th Percentile:   {np.percentile(edge_vals, 75):.3f} miles")
        print(f"    - Max Distance:      {np.max(edge_vals):.3f} miles")
        print(f"    - Mean +/- Std:      {np.mean(edge_vals):.3f} +/- {np.std(edge_vals):.3f} miles")
    print("-" * 80)


# =============================================================================
# 4. Publication-Quality Plotting
# =============================================================================

def plot_adjacency_overview(
    dist_matrix_zeros: np.ndarray,
    gaussian_matrix: np.ndarray,
    shortest_path_matrix: Optional[np.ndarray],
    edge_distances: np.ndarray,
    dataset_name: str,
    save_path: str = "distance_adjacency_matrix.png",
    show_gui: bool = False,
) -> None:
    """
    Renders a comprehensive multi-panel figure analyzing the distance adjacency structure.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    num_nodes = dist_matrix_zeros.shape[0]

    # Panel 1: Direct Edge Distance Adjacency Heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(dist_matrix_zeros, cmap="YlOrRd", interpolation="nearest")
    ax1.set_title(f"(A) Direct Distance Adjacency Matrix (Miles)\n{dataset_name} ({num_nodes} x {num_nodes} sensors)", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Target Sensor Index (v)", fontsize=10)
    ax1.set_ylabel("Source Sensor Index (u)", fontsize=10)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Distance (miles)", fontsize=9)

    # Panel 2: Thresholded Gaussian Kernel Matrix
    ax2 = axes[0, 1]
    im2 = ax2.imshow(gaussian_matrix, cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax2.set_title(f"(B) Thresholded Gaussian Kernel Adjacency Matrix\n$A_{{ij}} = \\exp(-d_{{ij}}^2 / \\sigma^2)$, $\\kappa=0.1$", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Target Sensor Index (j)", fontsize=10)
    ax2.set_ylabel("Source Sensor Index (i)", fontsize=10)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Spatial Correlation Weight [0, 1]", fontsize=9)

    # Panel 3: All-Pairs Shortest Path Distance Matrix
    ax3 = axes[1, 0]
    if shortest_path_matrix is not None:
        finite_sp = shortest_path_matrix.copy()
        finite_mask = ~np.isinf(finite_sp)
        if np.any(finite_mask):
            max_finite = np.max(finite_sp[finite_mask])
            finite_sp[np.isinf(finite_sp)] = max_finite * 1.15
        im3 = ax3.imshow(finite_sp, cmap="plasma", interpolation="nearest")
        ax3.set_title(f"(C) All-Pairs Shortest Path Distances\nNetworkX Floyd-Warshall Algorithm", fontsize=12, fontweight="bold", pad=8)
        ax3.set_xlabel("Destination Sensor Index", fontsize=10)
        ax3.set_ylabel("Origin Sensor Index", fontsize=10)
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label("Shortest Corridor Path (miles)", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Shortest Path Matrix Not Computed\n(Run with --mode shortest_path or --mode all)", ha="center", va="center")

    # Panel 4: Edge Distance Distribution Histogram
    ax4 = axes[1, 1]
    if len(edge_distances) > 0:
        counts, bins, patches = ax4.hist(
            edge_distances,
            bins=40,
            color="#2B6CB0",
            edgecolor="white",
            alpha=0.85,
            density=False,
        )
        mean_d = float(np.mean(edge_distances))
        median_d = float(np.median(edge_distances))
        ax4.axvline(mean_d, color="#E53E3E", linestyle="--", linewidth=2, label=f"Mean: {mean_d:.2f} mi")
        ax4.axvline(median_d, color="#38A169", linestyle="-.", linewidth=2, label=f"Median: {median_d:.2f} mi")
        ax4.set_title(f"(D) Physical Edge Distance Distribution\n({len(edge_distances):,} Directed Corridor Edges)", fontsize=12, fontweight="bold", pad=8)
        ax4.set_xlabel("Distance (miles)", fontsize=10)
        ax4.set_ylabel("Frequency (Edge Count)", fontsize=10)
        ax4.legend(frameon=True, facecolor="#F7FAFC", edgecolor="#CBD5E0")
        ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    print(_c(f"  [+] Saved high-resolution overview visualization to: {save_path}", BOLD + GREEN))

    if show_gui:
        try:
            plt.show()
        except Exception as e:
            print(f"  Note: GUI window could not be opened ({e}), image saved to disk.")


# =============================================================================
# 5. Main CLI Controller
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize the Distance Adjacency Matrix of a NetworkX Traffic Sensor Graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PEMS04",
        choices=["PEMS04", "PEMS08", "pems04", "pems08", "METR-LA", "metr-la", "metr_la"],
        help="Benchmark dataset to inspect.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Custom path to distance CSV file (overrides --dataset).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Base directory containing dataset subfolders.",
    )
    parser.add_argument(
        "--submatrix",
        type=int,
        default=10,
        help="Number of rows/cols (N x N) to display in the terminal table.",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default=None,
        help="Comma-separated list of specific node IDs to inspect (e.g., '0,2,14,16,25').",
    )
    parser.add_argument(
        "--nonedge",
        type=str,
        default="0.0",
        choices=["0.0", "inf", "nan"],
        help="Value assigned to non-adjacent node pairs in the distance matrix.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "direct", "gaussian", "shortest_path"],
        help="Adjacency representation to show in detail.",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Build an undirected Graph instead of a directed DiGraph.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip generating and saving the matplotlib visualization.",
    )
    parser.add_argument(
        "--save_fig",
        type=str,
        default="distance_adjacency_matrix.png",
        help="Output path for the heatmap visualization image.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Optional path to export the complete N x N distance adjacency matrix as CSV.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive matplotlib window in addition to saving the figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset_name = args.dataset.upper()

    # Resolve distance CSV path
    if args.file:
        distance_file = os.path.abspath(args.file)
    else:
        distance_file = os.path.abspath(os.path.join(args.data_dir, dataset_name, "distance.csv"))

    if not os.path.exists(distance_file):
        print(_c(f"Error: Distance file not found at: {distance_file}", RED), file=sys.stderr)
        print("Please ensure the dataset exists in the data directory or provide --file <path>.", file=sys.stderr)
        sys.exit(1)

    # 1. Build NetworkX Graph
    directed = not args.undirected
    G = build_networkx_graph(distance_file, directed=directed)
    print_graph_summary(G, dataset_name, distance_file)

    # Determine nodes to inspect
    all_nodes = sorted(G.nodes())
    if args.nodes:
        try:
            inspect_nodes = [int(x.strip()) for x in args.nodes.split(",") if x.strip()]
            invalid = [u for u in inspect_nodes if u not in G]
            if invalid:
                print(_c(f"Warning: Node(s) {invalid} not present in graph. Ignoring.", YELLOW))
            inspect_nodes = [u for u in inspect_nodes if u in G]
            if not inspect_nodes:
                inspect_nodes = all_nodes[:args.submatrix]
        except ValueError:
            print(_c("Error: Invalid node list format. Use comma-separated integers, e.g., '0,1,2,5'.", RED))
            inspect_nodes = all_nodes[:args.submatrix]
    else:
        inspect_nodes = all_nodes

    # Determine non-edge representation
    nonedge_map = {"0.0": 0.0, "inf": np.inf, "nan": np.nan}
    nonedge_val = nonedge_map[args.nonedge]

    # 2. Extract Distance Adjacency Matrix using NetworkX API
    dist_matrix = get_distance_adjacency_matrix(G, nodelist=all_nodes, nonedge_value=nonedge_val)
    dist_matrix_zeros = get_distance_adjacency_matrix(G, nodelist=all_nodes, nonedge_value=0.0)
    dist_matrix_inf = get_distance_adjacency_matrix(G, nodelist=all_nodes, nonedge_value=np.inf)

    # 3. Compute Gaussian Kernel Adjacency Matrix
    gaussian_matrix, sigma = get_gaussian_adjacency_matrix(dist_matrix_inf, normalized_k=0.1)

    # 4. Compute All-Pairs Shortest Path Matrix (if requested or in 'all' mode)
    shortest_path_mat: Optional[np.ndarray] = None
    if args.mode in ["all", "shortest_path"]:
        print(_c("  [*] Computing NetworkX All-Pairs Shortest Path Distance Matrix (Floyd-Warshall)...", CYAN))
        shortest_path_mat = get_shortest_path_matrix(G, nodelist=all_nodes)
        print(_c("  [+] Shortest path matrix computed successfully.", GREEN))

    # Print Diagnostics & Statistics
    print_matrix_statistics(dist_matrix_zeros, f"{dataset_name} Direct Edge Distances")

    # 5. Display Selected Submatrices in Terminal
    preview_k = min(args.submatrix, len(inspect_nodes))
    preview_nodes = inspect_nodes[:preview_k]

    # Submatrix indexing
    node_indices = [all_nodes.index(u) for u in preview_nodes]

    if args.mode in ["all", "direct"]:
        sub_dist = dist_matrix[np.ix_(node_indices, node_indices)]
        nonedge_str = "  inf  " if np.isinf(nonedge_val) else ("  nan  " if np.isnan(nonedge_val) else "   .   ")
        print_matrix_table(
            sub_dist,
            preview_nodes,
            title="1. Direct Edge Distance Adjacency Matrix [NetworkX nx.to_numpy_array(weight='distance')]",
            unit="miles",
            max_display=preview_k,
            nonedge_repr=nonedge_str,
        )

    if args.mode in ["all", "gaussian"]:
        sub_gauss = gaussian_matrix[np.ix_(node_indices, node_indices)]
        print_matrix_table(
            sub_gauss,
            preview_nodes,
            title=f"2. Thresholded Gaussian Kernel Spatial Correlation Matrix [sigma = {sigma:.2f} mi, kappa = 0.1]",
            unit="weight [0, 1]",
            max_display=preview_k,
            nonedge_repr="  0.0  ",
        )

    if args.mode in ["all", "shortest_path"] and shortest_path_mat is not None:
        sub_sp = shortest_path_mat[np.ix_(node_indices, node_indices)]
        print_matrix_table(
            sub_sp,
            preview_nodes,
            title="3. All-Pairs Shortest Path Matrix [NetworkX nx.floyd_warshall_numpy(weight='distance')]",
            unit="miles",
            max_display=preview_k,
            nonedge_repr="  inf  ",
        )

    # 6. Save Full Matrix to CSV (if requested)
    if args.save_csv:
        csv_path = os.path.abspath(args.save_csv)
        print(_c(f"  [*] Exporting full {len(all_nodes)}x{len(all_nodes)} distance matrix to CSV...", CYAN))
        df_export = pd.DataFrame(dist_matrix, index=all_nodes, columns=all_nodes)
        df_export.to_csv(csv_path)
        print(_c(f"  [+] Successfully saved matrix to: {csv_path}", BOLD + GREEN))

    # 7. Generate Overview Visualization
    if not args.no_plot:
        finite_edges = dist_matrix_zeros[dist_matrix_zeros > 0]
        plot_adjacency_overview(
            dist_matrix_zeros=dist_matrix_zeros,
            gaussian_matrix=gaussian_matrix,
            shortest_path_matrix=shortest_path_mat,
            edge_distances=finite_edges,
            dataset_name=dataset_name,
            save_path=args.save_fig,
            show_gui=args.interactive,
        )

    print("=" * 80)
    print(_c("  Analysis Complete! Distance Adjacency Matrix successfully extracted from NetworkX.", BOLD + GREEN))
    print("=" * 80)


if __name__ == "__main__":
    main()
