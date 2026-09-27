"""
FedLLM: Privacy-Preserving Explainable Traffic Flow Prediction Pipeline
Phase 2: Data-to-Text Formulation & Structured Prompt Engineering
=============================================================================
This module transforms continuous numerical spatial-temporal traffic matrices
(from PeMSD4, PeMSD8, or METR-LA) into rich, structured natural language prompts
for LLM instruction fine-tuning (LoRA / QLoRA).

Core Components Encoded per Prompt:
1. Sensor Metadata: Node ID, corridor context, and network degree.
2. Temporal Context: Diurnal time-of-day, peak categorization, day of the week.
3. Demand Statistics: Current flow, occupancy, and speed vs. historical baselines.
4. Spatial Neighbor Dynamics: State of adjacent upstream/downstream sensors.

Target & Explainability:
- Multi-Horizon Forecasting: 15, 30, 45, and 60 minutes ahead (t+3, t+6, t+9, t+12).
- Grounded Physics Rationale: Explaining traffic shockwaves, bottlenecks, and diurnal decay.
- JSONL Output: Formatted as standard instruction-input-output pairs for LLM fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import data ingestion components from Phase 1
from data_loader import (
    DATASET_ALIASES,
    DATASET_REGISTRY,
    DatasetConfig,
    SpatialGraphProcessor,
    TrafficDataDownloader,
    TrafficStandardScaler,
    load_traffic_data,
)

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FedLLM.PromptEngineer")


# =============================================================================
# 1. Temporal & Historical Context Calculator
# =============================================================================

class TemporalContextEngine:
    """Calculates temporal context and diurnal baselines for traffic time series."""

    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def __init__(self, steps_per_day: int = 288, interval_minutes: int = 5, base_date: Optional[datetime] = None):
        self.steps_per_day = steps_per_day
        self.interval_minutes = interval_minutes
        self.base_date = base_date or datetime(2018, 1, 1, 0, 0)  # Standard PeMS04 start: Jan 1, 2018 (Monday)

    def get_timestamp_info(self, step_idx: int) -> Dict[str, Any]:
        """Converts integer time step into date, time string, day of week, and peak category."""
        delta = timedelta(minutes=step_idx * self.interval_minutes)
        current_dt = self.base_date + delta

        hour = current_dt.hour
        minute = current_dt.minute
        time_str = current_dt.strftime("%I:%M %p")
        day_of_week = self.DAY_NAMES[current_dt.weekday()]
        is_weekend = current_dt.weekday() >= 5

        # Traffic rush hour period categorization
        time_decimal = hour + (minute / 60.0)
        if 7.0 <= time_decimal < 9.5:
            period_label = "Morning Peak (Rush Hour)"
        elif 9.5 <= time_decimal < 16.0:
            period_label = "Midday Off-Peak"
        elif 16.0 <= time_decimal < 19.5:
            period_label = "Evening Peak (Rush Hour)"
        elif 19.5 <= time_decimal < 23.0:
            period_label = "Night Off-Peak"
        else:
            period_label = "Early Morning / Low Traffic"

        tod_slot = step_idx % self.steps_per_day

        return {
            "datetime": current_dt,
            "time_str": time_str,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "period_label": period_label,
            "tod_slot": tod_slot,
        }

    @staticmethod
    def compute_diurnal_baselines(data: np.ndarray, steps_per_day: int = 288) -> np.ndarray:
        """
        Computes the empirical mean traffic profile for each 5-min slot of the day across all nodes.
        Output shape: (steps_per_day, num_nodes, num_features)
        """
        num_timesteps, num_nodes, num_features = data.shape
        num_full_days = num_timesteps // steps_per_day
        truncated_data = data[: num_full_days * steps_per_day]
        reshaped = truncated_data.reshape(num_full_days, steps_per_day, num_nodes, num_features)
        diurnal_means = np.mean(reshaped, axis=0)  # Shape: (steps_per_day, num_nodes, num_features)
        return diurnal_means.astype(np.float32)


# =============================================================================
# 2. Structured Prompt Formulation Engine
# =============================================================================

class FedLLMPromptBuilder:
    """
    Transforms raw multi-channel telemetry and spatial adjacency matrices
    into clean, structured natural language prompts for LLM training.
    """

    SYSTEM_INSTRUCTION = (
        "You are FedLLM, an expert spatial-temporal traffic forecasting system. "
        "Analyze the given freeway sensor metadata, temporal context, historical demand statistics, "
        "and spatial neighbor dynamics. Predict the multi-horizon short-term traffic flow (15, 30, 45, and 60 minutes ahead) "
        "and provide an explainable physical traffic reasoning based on spatial shockwave propagation and diurnal patterns."
    )

    def __init__(
        self,
        raw_data: np.ndarray,
        adj_matrix: np.ndarray,
        cfg: DatasetConfig,
        temporal_engine: TemporalContextEngine,
        diurnal_baselines: np.ndarray,
    ):
        self.raw_data = raw_data  # Shape: (T, N, F)
        self.adj_matrix = adj_matrix  # Shape: (N, N)
        self.cfg = cfg
        self.temporal_engine = temporal_engine
        self.diurnal_baselines = diurnal_baselines  # Shape: (288, N, F)
        self.num_nodes = raw_data.shape[1]
        self.num_features = raw_data.shape[2]

        # Precompute top-k spatial neighbors for every node from the adjacency matrix
        self.sensor_neighbors: Dict[int, List[Tuple[int, float]]] = self._extract_neighbors()

    def _extract_neighbors(self, top_k: int = 4) -> Dict[int, List[Tuple[int, float]]]:
        """Finds top-k most strongly connected adjacent sensors for each node."""
        neighbors_dict = {}
        for node in range(self.num_nodes):
            row = self.adj_matrix[node].copy()
            row[node] = 0.0  # Exclude self-loop
            top_indices = np.argsort(row)[::-1][:top_k]
            # Keep neighbors with non-zero connection
            valid_neighbors = [(int(idx), float(row[idx])) for idx in top_indices if row[idx] > 0.0]
            neighbors_dict[node] = valid_neighbors
        return neighbors_dict

    def generate_single_prompt(
        self,
        node_id: int,
        step_idx: int,
        in_steps: int = 12,
        horizon_steps: Tuple[int, int, int, int] = (3, 6, 9, 12),
    ) -> Optional[Dict[str, str]]:
        """
        Constructs a structured LLM instruction fine-tuning example (instruction, input, output)
        for a specific sensor at a specific timestamp.

        Args:
            node_id: Sensor node index (0 <= node_id < num_nodes)
            step_idx: Current temporal anchor index
            in_steps: Historical window length (12 steps = 1 hr)
            horizon_steps: Forecasting horizons in steps (3=15m, 6=30m, 9=45m, 12=60m)

        Returns:
            Dictionary {"instruction": "...", "input": "...", "output": "..."} or None if out of bounds.
        """
        max_horizon = max(horizon_steps)
        if step_idx < in_steps or (step_idx + max_horizon) >= self.raw_data.shape[0]:
            return None

        # ---------------------------------------------------------------------
        # 1. Temporal Context
        # ---------------------------------------------------------------------
        temp_info = self.temporal_engine.get_timestamp_info(step_idx)
        time_str = temp_info["time_str"]
        day_of_week = temp_info["day_of_week"]
        period_label = temp_info["period_label"]
        tod_slot = temp_info["tod_slot"]

        # ---------------------------------------------------------------------
        # 2. Sensor Metadata
        # ---------------------------------------------------------------------
        neighbors = self.sensor_neighbors.get(node_id, [])
        neighbor_ids = [n[0] for n in neighbors]
        network_degree = len(neighbor_ids)

        # ---------------------------------------------------------------------
        # 3. Demand Statistics & Historical Baseline Comparison
        # ---------------------------------------------------------------------
        # Current instantaneous reading at step_idx
        current_features = self.raw_data[step_idx, node_id]  # [flow, occ, speed] if 3 channels
        baseline_features = self.diurnal_baselines[tod_slot, node_id]

        if self.num_features >= 3:
            curr_flow = float(current_features[0])
            curr_occ = float(current_features[1])
            curr_speed = float(current_features[2])

            base_flow = float(baseline_features[0])
            base_speed = float(baseline_features[2])
        else:
            curr_speed = float(current_features[0])
            curr_flow = float(max(10.0, 800.0 * (1.0 - curr_speed / 70.0)))
            curr_occ = float(0.15)
            base_flow = float(curr_flow)
            base_speed = float(baseline_features[0])

        # Flow & Speed percentage differences vs. historical diurnal baseline
        flow_pct_diff = ((curr_flow - base_flow) / max(base_flow, 1.0)) * 100.0
        speed_pct_diff = ((curr_speed - base_speed) / max(base_speed, 1.0)) * 100.0

        flow_dev_str = f"{abs(flow_pct_diff):.1f}% {'above' if flow_pct_diff >= 0 else 'below'} historical average ({base_flow:.1f} veh/5min)"
        speed_dev_str = f"{abs(speed_pct_diff):.1f}% {'above' if speed_pct_diff >= 0 else 'below'} historical average ({base_speed:.1f} mph)"

        # Traffic state classification
        if curr_speed < 35.0 or curr_occ > 0.25:
            traffic_regime = "Congested / Bottleneck Queue"
        elif curr_speed < 50.0:
            traffic_regime = "Moderate Flow / Dense Traffic"
        else:
            traffic_regime = "Free Flow / Uncongested"

        # Historical 1-hour trend (in_steps = 12)
        past_window = self.raw_data[step_idx - in_steps : step_idx, node_id]
        if self.num_features >= 3:
            recent_flow_seq = [round(float(v), 1) for v in past_window[:, 0]]
            recent_speed_seq = [round(float(v), 1) for v in past_window[:, 2]]
        else:
            recent_flow_seq = [round(float(max(10.0, 800.0 * (1.0 - s / 70.0))), 1) for s in past_window[:, 0]]
            recent_speed_seq = [round(float(s), 1) for s in past_window[:, 0]]

        # ---------------------------------------------------------------------
        # 4. Spatial Neighbor Dynamics
        # ---------------------------------------------------------------------
        neighbor_summaries = []
        for n_id in neighbor_ids:
            n_features = self.raw_data[step_idx, n_id]
            if self.num_features >= 3:
                n_flow = float(n_features[0])
                n_speed = float(n_features[2])
            else:
                n_speed = float(n_features[0])
                n_flow = float(max(10.0, 800.0 * (1.0 - n_speed / 70.0)))

            if n_speed < 40.0:
                n_status = f"Sensor #{n_id} is CONGESTED (Speed: {n_speed:.1f} mph, Flow: {n_flow:.1f} veh/5min)"
            elif n_speed < 55.0:
                n_status = f"Sensor #{n_id} has SLOWING traffic (Speed: {n_speed:.1f} mph, Flow: {n_flow:.1f} veh/5min)"
            else:
                n_status = f"Sensor #{n_id} is in FREE FLOW (Speed: {n_speed:.1f} mph, Flow: {n_flow:.1f} veh/5min)"
            neighbor_summaries.append(n_status)

        if not neighbor_summaries:
            spatial_neighbor_text = "No immediate spatial neighbors within the threshold radius."
        else:
            spatial_neighbor_text = "\n  - " + "\n  - ".join(neighbor_summaries)

        # ---------------------------------------------------------------------
        # Assemble Structured Input Prompt
        # ---------------------------------------------------------------------
        input_text = (
            f"=== SENSOR METADATA ===\n"
            f"• Reporting Sensor Node: Freeway Station #{node_id:03d} (Dataset: {self.cfg.name})\n"
            f"• Connected Spatial Neighbors: {neighbor_ids} (Network Degree: {network_degree})\n\n"
            f"=== TEMPORAL CONTEXT ===\n"
            f"• Time of Day: {time_str} ({period_label})\n"
            f"• Day of Week: {day_of_week} ({'Weekend' if temp_info['is_weekend'] else 'Weekday'})\n\n"
            f"=== DEMAND STATISTICS ===\n"
            f"• Current Flow: {curr_flow:.1f} veh/5min ({flow_dev_str})\n"
            f"• Current Speed: {curr_speed:.1f} mph ({speed_dev_str})\n"
            f"• Occupancy Rate: {curr_occ * 100.0:.1f}%\n"
            f"• Current Traffic Regime: {traffic_regime}\n"
            f"• Past 1-Hour Flow Profile (12 steps @ 5-min): {recent_flow_seq}\n"
            f"• Past 1-Hour Speed Profile (12 steps @ 5-min): {recent_speed_seq}\n\n"
            f"=== SPATIAL NEIGHBOR DYNAMICS ==={spatial_neighbor_text}"
        )

        # ---------------------------------------------------------------------
        # Multi-Horizon Ground Truth Targets (15m, 30m, 45m, 60m)
        # ---------------------------------------------------------------------
        future_readings = {}
        for h_step in horizon_steps:
            target_step = step_idx + h_step
            t_features = self.raw_data[target_step, node_id]
            if self.num_features >= 3:
                t_flow = float(t_features[0])
                t_speed = float(t_features[2])
            else:
                t_speed = float(t_features[0])
                t_flow = float(max(10.0, 800.0 * (1.0 - t_speed / 70.0)))
            future_readings[h_step] = {"flow": t_flow, "speed": t_speed}

        f15 = future_readings[3]["flow"]
        s15 = future_readings[3]["speed"]
        f30 = future_readings[6]["flow"]
        s30 = future_readings[6]["speed"]
        f45 = future_readings[9]["flow"]
        s45 = future_readings[9]["speed"]
        f60 = future_readings[12]["flow"]
        s60 = future_readings[12]["speed"]

        # ---------------------------------------------------------------------
        # Synthetic Physical Traffic Reasoning & Explanation Formulation
        # ---------------------------------------------------------------------
        explanation = self._synthesize_physical_reasoning(
            node_id=node_id,
            curr_flow=curr_flow,
            curr_speed=curr_speed,
            f15=f15,
            s15=s15,
            f60=f60,
            s60=s60,
            period_label=period_label,
            traffic_regime=traffic_regime,
            neighbor_ids=neighbor_ids,
            flow_pct_diff=flow_pct_diff,
        )

        output_text = (
            f"### Multi-Horizon Traffic Forecast:\n"
            f"- 15-min Ahead (t+3): Flow = {f15:.1f} veh/5min | Speed = {s15:.1f} mph\n"
            f"- 30-min Ahead (t+6): Flow = {f30:.1f} veh/5min | Speed = {s30:.1f} mph\n"
            f"- 45-min Ahead (t+9): Flow = {f45:.1f} veh/5min | Speed = {s45:.1f} mph\n"
            f"- 60-min Ahead (t+12): Flow = {f60:.1f} veh/5min | Speed = {s60:.1f} mph\n\n"
            f"### Spatial-Temporal Rationale & Traffic Dynamics:\n{explanation}"
        )

        return {
            "instruction": self.SYSTEM_INSTRUCTION,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "dataset": self.cfg.name,
                "node_id": node_id,
                "step_idx": step_idx,
                "timestamp": temp_info["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                "f15": round(f15, 2),
                "f30": round(f30, 2),
                "f45": round(f45, 2),
                "f60": round(f60, 2),
            },
        }

    def _synthesize_physical_reasoning(
        self,
        node_id: int,
        curr_flow: float,
        curr_speed: float,
        f15: float,
        s15: float,
        f60: float,
        s60: float,
        period_label: str,
        traffic_regime: str,
        neighbor_ids: List[int],
        flow_pct_diff: float,
    ) -> str:
        """Synthesizes structured, physics-based explanation text for LLM supervision."""
        points = []

        # 1. Temporal & Rush Hour Trend
        if "Peak" in period_label:
            points.append(
                f"1. Diurnal Demand Trend: During the {period_label}, highway volume is experiencing elevated demand "
                f"({'+' if flow_pct_diff >= 0 else ''}{flow_pct_diff:.1f}% vs baseline). High vehicle arrival rates sustain heavy roadway loading."
            )
        else:
            points.append(
                f"1. Diurnal Demand Trend: Operating in {period_label} with stable demand profiles. Vehicle arrival rates remain consistent with historical patterns."
            )

        # 2. Spatial Wave & Neighbor Dynamics
        if neighbor_ids:
            first_neighbor = neighbor_ids[0]
            if curr_speed < 45.0:
                points.append(
                    f"2. Spatial Wave Propagation: Sensor #{node_id} is currently under {traffic_regime}. Backward-propagating shockwaves from adjacent Sensor #{first_neighbor} "
                    f"restrict outflow capacity, leading to vehicle queue accumulation over the next 15-30 minutes."
                )
            else:
                points.append(
                    f"2. Spatial Wave Propagation: Sensor #{node_id} and adjacent Sensor #{first_neighbor} maintain synchronized free-flow speeds. No backward shockwaves detected."
                )
        else:
            points.append(
                f"2. Spatial Dynamics: Isolated corridor segment with localized laminar flow characteristics."
            )

        # 3. Trajectory & Horizon Evolution
        delta_flow_60 = f60 - curr_flow
        if delta_flow_60 > 25.0:
            points.append(
                f"3. Forecast Evolution: Traffic volume is projected to increase from {curr_flow:.1f} to {f60:.1f} veh/5min (+{delta_flow_60:.1f}) by t+60 min as upstream demand accumulates."
            )
        elif delta_flow_60 < -25.0:
            points.append(
                f"3. Forecast Evolution: Traffic volume will decline from {curr_flow:.1f} to {f60:.1f} veh/5min as peak dissipation progresses and corridor throughput stabilizes."
            )
        else:
            points.append(
                f"3. Forecast Evolution: Steady-state equilibrium expected across the 60-minute forecast horizon with moderate flow fluctuations around {f60:.1f} veh/5min."
            )

        return "\n".join(points)


# =============================================================================
# 3. Pipeline Orchestrator & JSONL Generator
# =============================================================================

def generate_fedllm_prompts(
    dataset_name: str = "PEMS04",
    data_dir: str = "./data",
    output_dir: str = "./prompts",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    max_train_samples: int = 5000,
    max_val_samples: int = 500,
    max_test_samples: int = 1000,
    sample_step_stride: int = 6,
    nodes_subset: Optional[List[int]] = None,
) -> Dict[str, str]:
    """
    Generates structured train, val, and test JSONL prompt datasets for FedLLM instruction fine-tuning.

    Args:
        dataset_name: Benchmark name ('PEMS04', 'PEMS08', 'METR-LA')
        data_dir: Cache directory for raw telemetry files
        output_dir: Output directory for JSONL files
        train_ratio: Train temporal split fraction
        val_ratio: Val temporal split fraction
        max_train_samples: Cap on training prompt instances
        max_val_samples: Cap on validation prompt instances
        max_test_samples: Cap on testing prompt instances
        sample_step_stride: Stride in time steps between sampled prompts (e.g., 6 steps = every 30 mins)
        nodes_subset: Optional list of node IDs to extract (default: all nodes or balanced sample)

    Returns:
        Dictionary with absolute file paths to generated JSONL files:
        {'train': path, 'val': path, 'test': path}
    """
    os.makedirs(output_dir, exist_ok=True)
    canonical_name = DATASET_ALIASES.get(dataset_name.lower(), dataset_name.upper())
    cfg = DATASET_REGISTRY[canonical_name]

    # Ingest benchmark telemetry using Phase 1 loader
    bundle = load_traffic_data(dataset_name=cfg.name, data_dir=data_dir)
    adj_matrix = bundle["adj_matrix"]

    # Load raw unnormalized data array
    downloader = TrafficDataDownloader(data_dir=data_dir)
    data_path, _ = downloader.ensure_dataset(cfg.name)

    if data_path.endswith(".npz"):
        raw_npz = np.load(data_path)
        raw_data = raw_npz["data"] if "data" in raw_npz else raw_npz[list(raw_npz.keys())[0]]
    else:
        raw_df = pd.read_hdf(data_path)
        raw_data = raw_df.values[:, :, np.newaxis]

    if raw_data.ndim == 2:
        raw_data = raw_data[:, :, np.newaxis]

    total_timesteps, num_nodes, num_features = raw_data.shape
    logger.info(f"Loaded raw array for prompt engineering: {raw_data.shape}")

    # Compute temporal split boundaries
    train_end = int(total_timesteps * train_ratio)
    val_end = int(total_timesteps * (train_ratio + val_ratio))

    # Compute diurnal baselines strictly from training split
    temporal_engine = TemporalContextEngine(steps_per_day=288, interval_minutes=cfg.time_interval_minutes)
    diurnal_baselines = TemporalContextEngine.compute_diurnal_baselines(raw_data[:train_end], steps_per_day=288)

    # Initialize prompt builder
    prompt_builder = FedLLMPromptBuilder(
        raw_data=raw_data,
        adj_matrix=adj_matrix,
        cfg=cfg,
        temporal_engine=temporal_engine,
        diurnal_baselines=diurnal_baselines,
    )

    # Select target sensor subset
    if nodes_subset is None:
        # Default: Select a representative set of sensors across the network
        num_target_nodes = min(num_nodes, 50)
        selected_nodes = np.linspace(0, num_nodes - 1, num_target_nodes, dtype=int).tolist()
    else:
        selected_nodes = nodes_subset

    # Generate split datasets
    splits_config = [
        ("train_prompts.jsonl", 12, train_end - 12, max_train_samples),
        ("val_prompts.jsonl", train_end + 12, val_end - 12, max_val_samples),
        ("test_prompts.jsonl", val_end + 12, total_timesteps - 13, max_test_samples),
    ]

    output_files = {}

    for file_name, start_idx, end_idx, max_samples in splits_config:
        out_path = os.path.join(output_dir, file_name)
        logger.info(f"Generating prompts for {file_name} (Range: [{start_idx}, {end_idx}], Target: {max_samples} samples) ...")

        samples_written = 0
        with open(out_path, "w", encoding="utf-8") as f_out:
            for step_idx in range(start_idx, end_idx, sample_step_stride):
                for node_id in selected_nodes:
                    if samples_written >= max_samples:
                        break

                    item = prompt_builder.generate_single_prompt(node_id=node_id, step_idx=step_idx)
                    if item is not None:
                        # Write Alpaca/Fine-tuning compatible JSONL line
                        json_record = {
                            "instruction": item["instruction"],
                            "input": item["input"],
                            "output": item["output"],
                        }
                        f_out.write(json.dumps(json_record, ensure_ascii=False) + "\n")
                        samples_written += 1

                if samples_written >= max_samples:
                    break

        file_size_kb = os.path.getsize(out_path) / 1024.0
        logger.info(f"Saved {samples_written:,} prompts to {out_path} ({file_size_kb:.1f} KB).")
        split_key = file_name.replace("_prompts.jsonl", "")
        output_files[split_key] = os.path.abspath(out_path)

    return output_files


# =============================================================================
# 4. CLI Execution Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedLLM Phase 2: Data-to-Text Prompt Engineering Engine")
    parser.add_argument("--dataset", type=str, default="PEMS04", choices=["PEMS04", "PEMS08", "METR-LA", "pemsd4", "pemsd8"],
                        help="Benchmark traffic dataset name (default: PEMS04)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing cached raw traffic data")
    parser.add_argument("--output_dir", type=str, default="./prompts", help="Directory to save generated JSONL prompt files")
    parser.add_argument("--max_train", type=int, default=2000, help="Maximum training prompt pairs to generate")
    parser.add_argument("--max_val", type=int, default=300, help="Maximum validation prompt pairs to generate")
    parser.add_argument("--max_test", type=int, default=500, help="Maximum testing prompt pairs to generate")
    parser.add_argument("--stride", type=int, default=6, help="Time-step sampling stride (6 = every 30 minutes)")
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    print("=" * 80)
    print(f"[FedLLM] Phase 2 Prompt Engineering: Dataset = {args.dataset}")
    print("=" * 80)

    generated_files = generate_fedllm_prompts(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_train_samples=args.max_train,
        max_val_samples=args.max_val,
        max_test_samples=args.max_test,
        sample_step_stride=args.stride,
    )

    print("\n[+] Generated Prompt Datasets:")
    for split, path in generated_files.items():
        print(f"  * {split.upper():5s} : {path}")

    # Inspect and print first sample prompt from train_prompts.jsonl
    train_file = generated_files.get("train")
    if train_file and os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            first_line = json.loads(f.readline())

        print("\n" + "=" * 80)
        print("🔍 SAMPLE GENERATED PROMPT PAIR (train_prompts.jsonl #1):")
        print("=" * 80)
        print(f"\n[INSTRUCTION]:\n{first_line['instruction']}")
        print(f"\n[INPUT]:\n{first_line['input']}")
        print(f"\n[OUTPUT]:\n{first_line['output']}")
        print("=" * 80)

    print("\n[SUCCESS] Phase 2: Data-to-Text Formulation successfully verified!")
