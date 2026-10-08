"""
FedLLM: Dynamic Route Optimization Engine
=============================================================================
This module computes actionable, fastest freeway routes between sensor nodes
by leveraging FedLLM's multi-horizon predicted traffic speeds.

Core Architecture:
1. Spatial Graph Construction:
   - Builds a NetworkX directed graph (nx.DiGraph) from the PeMS sensor topology.
   - Nodes represent freeway loop detector stations.
   - Edges represent physical corridor distances (miles).

2. Dynamic Edge Weighting:
   - Evaluates FedLLM predicted speeds (e.g. t+15 min horizon) per sensor node.
   - Dynamic Cost Formula:
         Travel_Time (hours) = Distance (miles) / Predicted_Speed (mph)
         Travel_Time (minutes) = (Distance / Predicted_Speed) * 60.0
   - Dynamically reweights graph edges with predicted travel times.

3. Routing Algorithm:
   - get_fastest_route(): Computes the global fastest trajectory via Dijkstra's
     algorithm weighted by predicted travel times.
   - Compares against static distance-based routing to quantify time savings
     achieved by proactively bypassing predicted bottlenecks.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# ── Module Logger ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FedLLM.RouteOptimizer")

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
    """Wrap text in ANSI color escape when stdout supports it."""
    if sys.stdout.isatty() or os.environ.get("FORCE_COLOR"):
        return f"{colour}{text}{RESET}"
    return text


# =============================================================================
# 1. Data Structures
# =============================================================================

@dataclass
class SegmentInfo:
    """Details for a single edge/leg between two consecutive sensors."""
    from_node: int
    to_node: int
    distance_miles: float
    predicted_speed_mph: float
    travel_time_minutes: float
    regime: str  # "Free Flow", "Moderate", "Congested"


@dataclass
class RouteResult:
    """Full routing trajectory results and summary statistics."""
    source_node: int
    target_node: int
    path: List[int]
    total_distance_miles: float
    total_time_minutes: float
    average_speed_mph: float
    segments: List[SegmentInfo] = field(default_factory=list)
    routing_method: str = "FedLLM Dynamic (Travel Time)"


# =============================================================================
# 2. Route Optimizer Engine
# =============================================================================

class FedLLMRouteOptimizer:
    """
    Constructs the spatial sensor graph and executes dynamic route optimization
    driven by FedLLM multi-horizon speed forecasts.
    """

    MIN_SPEED_MPH = 5.0   # Clamp to prevent division by zero during complete gridlock
    FREE_FLOW_MPH = 50.0  # Threshold for free-flow regime
    CONGESTION_MPH = 35.0 # Threshold for congested regime

    def __init__(
        self,
        dataset_name: str = "PEMS04",
        data_dir: str = "./data",
        distance_file: Optional[str] = None,
    ):
        self.dataset_name = dataset_name.upper()
        self.data_dir = os.path.abspath(data_dir)

        # Resolve directory name (handle SZ-taxi, Porto, METR-LA, etc.)
        candidate_folders = [
            dataset_name,
            self.dataset_name,
            self.dataset_name.lower(),
            self.dataset_name.capitalize(),
            "SZ-taxi" if self.dataset_name in ["SZ-TAXI", "SZ_TAXI", "SZ-TAXI"] else dataset_name,
            "Porto" if self.dataset_name == "PORTO" else dataset_name,
            "METR-LA" if self.dataset_name == "METR-LA" else dataset_name,
        ]
        self.folder_path = os.path.join(self.data_dir, self.dataset_name)
        for cand in candidate_folders:
            p = os.path.join(self.data_dir, cand)
            if os.path.exists(p):
                self.folder_path = p
                break

        # Resolve distance file path
        if distance_file is None:
            candidate_dist_files = [
                os.path.join(self.folder_path, "distance.csv"),
                os.path.join(self.folder_path, "distances_la_2012.csv"),
            ]
            self.distance_file = os.path.join(self.folder_path, "distance.csv")
            for df in candidate_dist_files:
                if os.path.exists(df):
                    self.distance_file = df
                    break
        else:
            self.distance_file = os.path.abspath(distance_file)

        # Build underlying NetworkX directed graph
        self.graph: nx.DiGraph = self.build_sensor_graph(self.distance_file)
        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()

        # Cache of latest predicted speeds {node_id: speed_mph}
        self.current_speeds: Dict[int, float] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Graph Construction
    # ─────────────────────────────────────────────────────────────────────────
    def build_sensor_graph(self, distance_file: str) -> nx.DiGraph:
        """
        Loads the physical distance matrix and constructs a NetworkX directed graph.

        Each edge stores:
          - distance: physical segment distance in miles.
          - travel_time: initial travel time based on 60 mph free-flow.
        """
        if not os.path.exists(distance_file):
            raise FileNotFoundError(f"Distance file not found: {distance_file}")

        dist_df = pd.read_csv(distance_file)

        # Standardize column headers
        col_map = {}
        for col in dist_df.columns:
            clean = str(col).strip().lower()
            if clean in ["from", "src", "start", "0"]:
                col_map[col] = "from"
            elif clean in ["to", "dst", "end", "1"]:
                col_map[col] = "to"
            elif clean in ["cost", "distance", "dist", "weight", "2"]:
                col_map[col] = "cost"
        dist_df = dist_df.rename(columns=col_map)

        graph = nx.DiGraph()

        # Add directed edges with physical distance
        for _, row in dist_df.iterrows():
            u = int(row["from"])
            v = int(row["to"])
            distance_miles = float(row["cost"])

            # Default free-flow initial travel time (assuming 60 mph)
            initial_time_minutes = (distance_miles / 60.0) * 60.0

            graph.add_edge(
                u,
                v,
                distance=distance_miles,
                speed=60.0,
                travel_time=initial_time_minutes,
            )

        logger.info(
            f"Loaded {self.dataset_name} Sensor Graph: "
            f"{graph.number_of_nodes()} stations, {graph.number_of_edges()} corridor edges."
        )
        return graph

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Query FedLLM Predicted Speeds
    # ─────────────────────────────────────────────────────────────────────────
    def query_fedllm_predicted_speeds(
        self,
        step_idx: Optional[int] = None,
        horizon: str = "15min",
        synthetic_bottlenecks: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        """
        Queries the FedLLM inference engine / telemetry tensor to obtain
        predicted speeds for all sensor nodes at the target forecasting horizon.

        Args:
            step_idx: Time step index in the dataset (defaults to test split).
            horizon: Forecast window ("15min", "30min", "45min", "60min").
            synthetic_bottlenecks: Optional dict of {node_id: forced_speed} to
                simulate localized shockwaves / accidents for demonstration.

        Returns:
            Dictionary mapping {node_id: predicted_speed_mph}.
        """
        speeds: Dict[int, float] = {}

        # Attempt to load raw telemetry tensor for authentic base values
        candidate_npz = [
            os.path.join(self.folder_path, f"{self.dataset_name}.npz"),
            os.path.join(self.folder_path, f"{self.dataset_name.lower().replace('-', '_')}.npz"),
            os.path.join(self.folder_path, f"{self.dataset_name.lower()}.npz"),
        ]
        npz_path = None
        for cp in candidate_npz:
            if os.path.exists(cp):
                npz_path = cp
                break
        if npz_path is None and os.path.exists(self.folder_path):
            for f in os.listdir(self.folder_path):
                if f.endswith(".npz"):
                    npz_path = os.path.join(self.folder_path, f)
                    break

        if npz_path and os.path.exists(npz_path):
            try:
                npz_loaded = np.load(npz_path)
                raw_data = npz_loaded["data"] if "data" in npz_loaded else npz_loaded[list(npz_loaded.keys())[0]]
                # Default step: test split start
                target_step = step_idx if step_idx is not None else min(100, raw_data.shape[0] - 13)

                # Horizon offset depending on sampling interval (5-min vs 15-min)
                if raw_data.shape[0] < 5000:
                    # 15-min interval datasets (e.g. SZ-taxi, Porto)
                    horizon_offset_map = {"15min": 1, "30min": 2, "45min": 3, "60min": 4}
                else:
                    # 5-min interval datasets (e.g. PeMS, METR-LA)
                    horizon_offset_map = {"15min": 3, "30min": 6, "45min": 9, "60min": 12}

                offset = horizon_offset_map.get(horizon.lower(), 1 if raw_data.shape[0] < 5000 else 3)
                eval_step = min(target_step + offset, raw_data.shape[0] - 1)

                num_nodes_in_data = raw_data.shape[1]
                for node in self.graph.nodes():
                    if node < num_nodes_in_data:
                        if raw_data.ndim == 3:
                            spd_idx = 2 if raw_data.shape[2] >= 3 else 0
                            base_spd = float(raw_data[eval_step, node, spd_idx])
                        elif raw_data.ndim == 2:
                            base_spd = float(raw_data[eval_step, node])
                        else:
                            base_spd = 50.0
                    else:
                        base_spd = 50.0
                    speeds[node] = max(self.MIN_SPEED_MPH, base_spd)
            except Exception as e:
                logger.warning(f"Could not read speed from npz ({e}), using default baseline.")
                for node in self.graph.nodes():
                    speeds[node] = 50.0
        else:
            for node in self.graph.nodes():
                speeds[node] = 50.0

        # Inject user-defined bottlenecks (e.g., accidents, construction)
        if synthetic_bottlenecks:
            for node, forced_speed in synthetic_bottlenecks.items():
                if node in self.graph:
                    speeds[node] = max(self.MIN_SPEED_MPH, float(forced_speed))

        self.current_speeds = speeds
        return speeds

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Dynamic Edge Weighting (The Core Logic)
    # ─────────────────────────────────────────────────────────────────────────
    def update_edge_weights_with_predictions(
        self,
        predicted_speeds: Dict[int, float],
    ) -> None:
        """
        Dynamically updates every edge weight in the network graph using:
            Travel_Time (hours) = Distance (miles) / Predicted_Speed (mph)
            Travel_Time (minutes) = (Distance / Predicted_Speed) * 60.0

        The effective segment speed for edge (u -> v) is modeled as the average
        of the predicted speeds at source sensor u and destination sensor v.
        """
        for u, v, data in self.graph.edges(data=True):
            dist_miles = data["distance"]

            speed_u = predicted_speeds.get(u, 55.0)
            speed_v = predicted_speeds.get(v, 55.0)

            # Average speed across the corridor segment
            effective_speed = max(self.MIN_SPEED_MPH, (speed_u + speed_v) / 2.0)

            # Dynamic travel time formula (minutes)
            travel_time_minutes = (dist_miles / effective_speed) * 60.0

            # Store updated dynamic weights
            data["speed"] = effective_speed
            data["travel_time"] = travel_time_minutes

        logger.info(
            f"Dynamic edge weights updated for {self.graph.number_of_edges()} edges "
            f"using FedLLM predicted traffic speeds."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Routing Algorithms (Dijkstra)
    # ─────────────────────────────────────────────────────────────────────────
    def get_fastest_route(
        self,
        source_node: int,
        target_node: int,
    ) -> RouteResult:
        """
        Finds the optimal fastest route using Dijkstra's algorithm weighted
        by FedLLM predicted travel times.
        """
        return self._compute_route(
            source_node=source_node,
            target_node=target_node,
            weight_attribute="travel_time",
            method_name="FedLLM Dynamic (Predicted Travel Time)",
        )

    def get_shortest_distance_route(
        self,
        source_node: int,
        target_node: int,
    ) -> RouteResult:
        """
        Finds the standard shortest-distance route (static navigation baseline)
        weighted purely by physical miles, then evaluates its actual travel time
        under current traffic conditions.
        """
        return self._compute_route(
            source_node=source_node,
            target_node=target_node,
            weight_attribute="distance",
            method_name="Static Shortest Distance (Baseline)",
        )

    def _compute_route(
        self,
        source_node: int,
        target_node: int,
        weight_attribute: str,
        method_name: str,
    ) -> RouteResult:
        """Internal helper to execute Dijkstra pathfinding and assemble metrics."""
        if source_node not in self.graph:
            raise ValueError(f"Source sensor #{source_node} does not exist in graph.")
        if target_node not in self.graph:
            raise ValueError(f"Target sensor #{target_node} does not exist in graph.")
        if source_node == target_node:
            return RouteResult(
                source_node=source_node,
                target_node=target_node,
                path=[source_node],
                total_distance_miles=0.0,
                total_time_minutes=0.0,
                average_speed_mph=0.0,
                segments=[],
                routing_method=method_name,
            )

        # Run Dijkstra's algorithm via NetworkX
        try:
            path = nx.shortest_path(
                self.graph,
                source=source_node,
                target=target_node,
                weight=weight_attribute,
            )
        except nx.NetworkXNoPath:
            raise nx.NetworkXNoPath(
                f"No directed freeway path exists from Station #{source_node} to Station #{target_node}."
            )

        # Accumulate segment details
        total_dist = 0.0
        total_time = 0.0
        segments: List[SegmentInfo] = []

        for u, v in zip(path[:-1], path[1:]):
            edge_data = self.graph[u][v]
            d = edge_data["distance"]
            spd = edge_data.get("speed", 55.0)
            # Recompute travel time with edge's latest speed
            t = (d / spd) * 60.0

            total_dist += d
            total_time += t

            # Categorize regime
            if spd < self.CONGESTION_MPH:
                regime = "Congested"
            elif spd < self.FREE_FLOW_MPH:
                regime = "Moderate"
            else:
                regime = "Free Flow"

            segments.append(
                SegmentInfo(
                    from_node=u,
                    to_node=v,
                    distance_miles=d,
                    predicted_speed_mph=spd,
                    travel_time_minutes=t,
                    regime=regime,
                )
            )

        avg_speed = (total_dist / (total_time / 60.0)) if total_time > 0 else 0.0

        return RouteResult(
            source_node=source_node,
            target_node=target_node,
            path=path,
            total_distance_miles=total_dist,
            total_time_minutes=total_time,
            average_speed_mph=avg_speed,
            segments=segments,
            routing_method=method_name,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Route Comparison (Dynamic vs. Static Baseline)
    # ─────────────────────────────────────────────────────────────────────────
    def compare_routes(
        self,
        source_node: int,
        target_node: int,
    ) -> Dict[str, Any]:
        """
        Computes both the dynamic fastest route and the static shortest route,
        calculating time saved, distance difference, and speed advantages.
        """
        dynamic_route = self.get_fastest_route(source_node, target_node)
        static_route = self.get_shortest_distance_route(source_node, target_node)

        time_saved_min = static_route.total_time_minutes - dynamic_route.total_time_minutes
        time_saved_pct = (
            (time_saved_min / static_route.total_time_minutes) * 100.0
            if static_route.total_time_minutes > 0
            else 0.0
        )
        distance_diff_miles = dynamic_route.total_distance_miles - static_route.total_distance_miles

        return {
            "dynamic_route": dynamic_route,
            "static_route": static_route,
            "time_saved_minutes": time_saved_min,
            "time_saved_percent": time_saved_pct,
            "distance_diff_miles": distance_diff_miles,
            "bypassed_congestion": time_saved_min > 0.5,
        }


# =============================================================================
# 3. Clean Display / Formatting Utilities
# =============================================================================

def print_route_dashboard(
    optimizer: FedLLMRouteOptimizer,
    source_node: int,
    target_node: int,
    comparison: Dict[str, Any],
) -> None:
    """Prints a beautiful, professional ITS route optimization terminal report."""
    dyn: RouteResult = comparison["dynamic_route"]
    stat: RouteResult = comparison["static_route"]
    saved_min = comparison["time_saved_minutes"]
    saved_pct = comparison["time_saved_percent"]

    width = 82
    divider = "═" * width
    sub_divider = "─" * width

    print()
    print(_c(divider, CYAN))
    print(_c(f"  🚦 FedLLM: DYNAMIC FREEWAY ROUTE OPTIMIZATION ({optimizer.dataset_name})", BOLD + CYAN))
    print(_c(f"  Origin: Station #{source_node:03d}  ──▶  Destination: Station #{target_node:03d}", CYAN))
    print(_c(divider, CYAN))

    # ── Summary Cards ─────────────────────────────────────────────────────────
    print()
    print(_c("  [1] ROUTE COMPARISON SUMMARY", BOLD + YELLOW))
    print(sub_divider)

    row_fmt = "  {:<26} {:<24} {:<24}"
    print(_c(row_fmt.format("Metric", "FedLLM Dynamic (Fastest)", "Static Distance-Only"), BOLD))
    print(row_fmt.format("─" * 26, "─" * 24, "─" * 24))

    # Path sequence string
    dyn_path_str = " ➔ ".join(f"#{n}" for n in dyn.path)
    stat_path_str = " ➔ ".join(f"#{n}" for n in stat.path)

    print(row_fmt.format("Total Travel Time", f"{_c(f'{dyn.total_time_minutes:.1f} mins', GREEN + BOLD)}", f"{stat.total_time_minutes:.1f} mins"))
    print(row_fmt.format("Total Distance", f"{dyn.total_distance_miles:.2f} miles", f"{stat.total_distance_miles:.2f} miles"))
    print(row_fmt.format("Average Speed", f"{dyn.average_speed_mph:.1f} mph", f"{stat.average_speed_mph:.1f} mph"))
    print(row_fmt.format("Number of Hops", f"{len(dyn.path) - 1} segments", f"{len(stat.path) - 1} segments"))

    # Savings Callout
    print()
    if saved_min > 0.5:
        savings_msg = (
            f"  ⚡ TIME SAVED: {saved_min:.1f} minutes ({saved_pct:.1f}% faster) "
            f"by proactively routing around predicted bottlenecks!"
        )
        print(_c(savings_msg, GREEN + BOLD))
    else:
        print(_c("  ℹ️ Both methods chose the same route — corridor is currently in free-flow equilibrium.", BLUE))

    # ── Optimal Trajectory Path ───────────────────────────────────────────────
    print()
    print(_c("  [2] OPTIMAL SEQUENCE OF SENSORS (FedLLM Trajectory)", BOLD + GREEN))
    print(sub_divider)
    print(f"  {_c(dyn_path_str, BOLD + GREEN)}")
    print()

    # ── Segment Breakdown Table ───────────────────────────────────────────────
    print(_c("  [3] TURN-BY-TURN CORRIDOR TELEMETRY", BOLD + MAGENTA))
    print(sub_divider)
    seg_header = "  {:<6} {:<16} {:<12} {:<18} {:<16} {:<12}"
    print(_c(seg_header.format("Leg", "Corridor Segment", "Distance", "Predicted Speed", "Travel Time", "Regime"), BOLD))
    print(seg_header.format("─" * 6, "─" * 16, "─" * 12, "─" * 18, "─" * 16, "─" * 12))

    for idx, seg in enumerate(dyn.segments, 1):
        seg_name = f"#{seg.from_node:03d} ➔ #{seg.to_node:03d}"
        if seg.regime == "Free Flow":
            regime_badge = _c("🟢 Free Flow", GREEN)
        elif seg.regime == "Moderate":
            regime_badge = _c("🟡 Moderate", YELLOW)
        else:
            regime_badge = _c("🔴 Congested", RED)

        print(
            seg_header.format(
                f"#{idx:02d}",
                seg_name,
                f"{seg.distance_miles:.2f} mi",
                f"{seg.predicted_speed_mph:.1f} mph",
                f"{seg.travel_time_minutes:.1f} min",
                regime_badge,
            )
        )

    print()
    print(_c(divider, CYAN))


# =============================================================================
# 4. Command-Line Interface (CLI)
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedLLM: Dynamic Route Optimization based on Predicted Traffic Speeds"
    )
    parser.add_argument("--source", type=int, default=0,
                        help="Origin sensor station ID (default: 0)")
    parser.add_argument("--target", type=int, default=100,
                        help="Destination sensor station ID (default: 100)")
    parser.add_argument("--dataset", type=str, default="PEMS04",
                        choices=["PEMS04", "PEMS08"],
                        help="Freeway dataset network (default: PEMS04)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to datasets directory")
    parser.add_argument("--horizon", type=str, default="15min",
                        choices=["15min", "30min", "45min", "60min"],
                        help="FedLLM forecast horizon to optimize against (default: 15min)")
    parser.add_argument("--simulate_bottleneck", action="store_true",
                        help="Optionally inject a synthetic traffic incident (default: False)")
    parser.add_argument("--bottleneck_node", type=int, default=96,
                        help="Sensor node ID for optional synthetic bottleneck (default: 96)")
    parser.add_argument("--bottleneck_speed", type=float, default=12.0,
                        help="Speed in mph for optional bottleneck (default: 12.0 mph)")
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    # Initialize optimizer
    optimizer = FedLLMRouteOptimizer(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
    )

    # Optional synthetic bottleneck (off by default)
    bottlenecks = None
    if args.simulate_bottleneck:
        bottlenecks = {args.bottleneck_node: args.bottleneck_speed}

    # Step 1: Query FedLLM predicted speeds
    predicted_speeds = optimizer.query_fedllm_predicted_speeds(
        horizon=args.horizon,
        synthetic_bottlenecks=bottlenecks,
    )

    # Step 2: Dynamic Edge Weighting
    optimizer.update_edge_weights_with_predictions(predicted_speeds)

    # Step 3: Run Route Optimization & Comparison
    try:
        comparison = optimizer.compare_routes(
            source_node=args.source,
            target_node=args.target,
        )
        # Step 4: Display Rich Output
        print_route_dashboard(
            optimizer=optimizer,
            source_node=args.source,
            target_node=args.target,
            comparison=comparison,
        )
    except nx.NetworkXNoPath as e:
        print(_c(f"\n[!] Routing Error: {e}", RED))
        sys.exit(1)
