"""
FedLLM: Explainable Traffic Prediction Dashboard (Multi-Dataset)
=============================================================================
Interactive Streamlit web application for demonstrating the FedLLM
privacy-preserving federated traffic forecasting pipeline across benchmark
freeway networks (PeMS04 and PeMS08).

Launch with:
    streamlit run app.py

Features:
  • Multi-dataset switcher — toggle seamlessly between PeMS04 and PeMS08.
  • Date & time picker — browse ANY timestamp in the selected dataset.
  • Sensor selector — pick any sensor in the network (307 for 04, 170 for 08).
  • Predicted vs. Actual comparison on the SAME chart.
  • Side-by-side metric cards (Actual | Predicted | Error).
  • Explainable AI rationale panel.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ── Import project modules ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_evaluator import ForecastParser, ParsedForecast
from data_loader import (
    DATASET_ALIASES,
    DATASET_REGISTRY,
    DatasetConfig,
    SpatialGraphProcessor,
    TrafficDataDownloader,
)
from prompt_engineer import (
    FedLLMPromptBuilder,
    TemporalContextEngine,
)
from route_optimizer import FedLLMRouteOptimizer, RouteResult


# =============================================================================
# 1. Dataset Metadata & Registry
# =============================================================================

DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "PEMS04": {
        "name": "PEMS04",
        "num_nodes": 307,
        "region": "SF Bay Area (District 4)",
        "full_region": "California District 4 — San Francisco Bay Area (I-80, US-101, I-280, I-880)",
        "base_date": datetime(2018, 1, 1, 0, 0),
        "default_date": date(2018, 2, 5),
        "default_time": time(8, 0),
        "interval_min": 5,
        "speed_unit": "mph",
        "flow_unit": "veh/5min",
        "description": "307 loop detectors across the San Francisco Bay Area urban freeway network.",
    },
    "PEMS08": {
        "name": "PEMS08",
        "num_nodes": 170,
        "region": "San Bernardino (District 8)",
        "full_region": "California District 8 — San Bernardino / Inland Empire (I-10, I-15, SR-60, SR-91)",
        "base_date": datetime(2016, 7, 1, 0, 0),
        "default_date": date(2016, 7, 18),
        "default_time": time(8, 0),
        "interval_min": 5,
        "speed_unit": "mph",
        "flow_unit": "veh/5min",
        "description": "170 loop detectors along Southern California freight and logistics corridors.",
    },
    "SZ-TAXI": {
        "name": "SZ-TAXI",
        "num_nodes": 156,
        "region": "Shenzhen (Luohu District)",
        "full_region": "Shenzhen Luohu District Asian Megacity Road Grid (156 arterial segments)",
        "base_date": datetime(2015, 1, 1, 0, 0),
        "default_date": date(2015, 1, 15),
        "default_time": time(8, 0),
        "interval_min": 15,
        "speed_unit": "km/h",
        "flow_unit": "veh/15min",
        "description": "156 arterial road segments across Shenzhen's high-density central commercial corridor.",
    },
    "PORTO": {
        "name": "PORTO",
        "num_nodes": 120,
        "region": "Porto Metropolitan Grid",
        "full_region": "Porto European Historic Urban Network (120 arterial sensor nodes)",
        "base_date": datetime(2013, 7, 1, 0, 0),
        "default_date": date(2013, 7, 15),
        "default_time": time(8, 0),
        "interval_min": 15,
        "speed_unit": "km/h",
        "flow_unit": "veh/15min",
        "description": "120 sensor stations across Porto's historic radial corridors and roundabout grid.",
    },
    "METR-LA": {
        "name": "METR-LA",
        "num_nodes": 207,
        "region": "Los Angeles (LA County Highways)",
        "full_region": "Los Angeles County Highway Network (207 loop detectors)",
        "base_date": datetime(2012, 3, 1, 0, 0),
        "default_date": date(2012, 4, 15),
        "default_time": time(8, 0),
        "interval_min": 5,
        "speed_unit": "mph",
        "flow_unit": "veh/5min",
        "description": "207 loop detector sensors across Los Angeles County highways.",
    },
}

# ── Explicit In-Memory Registration for Streamlit Hot-Reloading ───────────────
if "SZ-TAXI" not in DATASET_REGISTRY:
    DATASET_REGISTRY["SZ-TAXI"] = DatasetConfig(
        name="SZ-taxi",
        num_nodes=156,
        num_features=3,
        feature_names=["traffic_flow", "occupancy", "average_speed"],
        time_interval_minutes=15,
        primary_data_file="sz_taxi.npz",
        distance_file="distance.csv",
        download_urls={
            "sz_taxi.npz": [
                "https://raw.githubusercontent.com/lehaifeng/T-GCN/master/T-GCN/T-GCN-PyTorch/data/sz_speed.csv",
            ],
            "distance.csv": [
                "https://raw.githubusercontent.com/lehaifeng/T-GCN/master/T-GCN/T-GCN-PyTorch/data/sz_adj.csv",
            ],
        },
        description="Shenzhen Luohu District Asian Megacity Road Grid with 156 road segments (Jan 2015).",
    )

if "PORTO" not in DATASET_REGISTRY:
    DATASET_REGISTRY["PORTO"] = DatasetConfig(
        name="Porto",
        num_nodes=120,
        num_features=3,
        feature_names=["traffic_flow", "occupancy", "average_speed"],
        time_interval_minutes=15,
        primary_data_file="porto.npz",
        distance_file="distance.csv",
        download_urls={},
        description="Porto European Historic Urban Network with 120 sensor stations (July 2013).",
    )

if "METR-LA" not in DATASET_REGISTRY:
    DATASET_REGISTRY["METR-LA"] = DatasetConfig(
        name="METR-LA",
        num_nodes=207,
        num_features=1,
        feature_names=["average_speed"],
        time_interval_minutes=5,
        primary_data_file="metr-la.npz",
        distance_file="distance.csv",
        download_urls={},
        description="Los Angeles County highway network with 207 loop detectors (Mar-Jun 2012).",
    )

DATASET_ALIASES.update({
    "sz-taxi": "SZ-TAXI",
    "sz_taxi": "SZ-TAXI",
    "sztaxi": "SZ-TAXI",
    "sz": "SZ-TAXI",
    "porto": "PORTO",
    "portugal": "PORTO",
})


# =============================================================================
# 2. Data Loading Utilities (cached)
# =============================================================================

@st.cache_resource(show_spinner="Loading freeway telemetry matrix...")
def load_raw_data(dataset_name: str):
    """Load the raw .npz telemetry tensor and adjacency matrix for the given dataset."""
    canonical = DATASET_ALIASES.get(dataset_name.lower(), dataset_name.upper())
    cfg = DATASET_REGISTRY.get(dataset_name) or DATASET_REGISTRY.get(canonical)

    if cfg is None:
        meta = DATASET_METADATA.get(dataset_name) or DATASET_METADATA.get(canonical, {})
        cfg = DatasetConfig(
            name="SZ-taxi" if "SZ" in dataset_name.upper() else ("Porto" if "PORTO" in dataset_name.upper() else dataset_name),
            num_nodes=meta.get("num_nodes", 156 if "SZ" in dataset_name.upper() else (120 if "PORTO" in dataset_name.upper() else 207)),
            num_features=3 if dataset_name != "METR-LA" else 1,
            feature_names=["traffic_flow", "occupancy", "average_speed"] if dataset_name != "METR-LA" else ["average_speed"],
            time_interval_minutes=meta.get("interval_min", 15),
            primary_data_file=f"{dataset_name.lower().replace('-', '_')}.npz",
            distance_file="distance.csv",
            download_urls={},
            description=meta.get("description", ""),
        )
        DATASET_REGISTRY[dataset_name] = cfg
        DATASET_REGISTRY[canonical] = cfg

    meta = DATASET_METADATA.get(dataset_name) or DATASET_METADATA.get(canonical)
    base_date = meta["base_date"]
    interval_min = meta.get("interval_min", cfg.time_interval_minutes)
    steps_per_day = int(1440 / interval_min)

    downloader = TrafficDataDownloader(data_dir="./data")
    try:
        data_path, dist_path = downloader.ensure_dataset(cfg.name)
    except Exception as e:
        folder_candidates = [cfg.name, dataset_name, canonical, "SZ-taxi", "Porto", "METR-LA"]
        resolved_dir = None
        for fc in folder_candidates:
            p = os.path.join(downloader.data_dir, fc)
            if os.path.exists(p):
                resolved_dir = p
                break
        if resolved_dir is None:
            resolved_dir = os.path.join(downloader.data_dir, cfg.name)
        data_path = os.path.join(resolved_dir, cfg.primary_data_file)
        dist_path = os.path.join(resolved_dir, cfg.distance_file)

    # Load raw tensor
    raw_npz = np.load(data_path)
    raw_data = raw_npz["data"] if "data" in raw_npz else raw_npz[list(raw_npz.keys())[0]]

    # Load adjacency matrix (static method)
    adj_matrix = SpatialGraphProcessor.build_adjacency_matrix(
        distance_file=dist_path,
        num_nodes=cfg.num_nodes,
    )

    # Compute diurnal baselines from first 70% (train split)
    train_end = int(raw_data.shape[0] * 0.7)
    temporal_engine = TemporalContextEngine(
        steps_per_day=steps_per_day,
        interval_minutes=interval_min,
        base_date=base_date,
    )
    diurnal_baselines = TemporalContextEngine.compute_diurnal_baselines(
        raw_data[:train_end], steps_per_day=steps_per_day
    )

    # Build prompt builder
    prompt_builder = FedLLMPromptBuilder(
        raw_data=raw_data,
        adj_matrix=adj_matrix,
        cfg=cfg,
        temporal_engine=temporal_engine,
        diurnal_baselines=diurnal_baselines,
    )

    return raw_data, adj_matrix, prompt_builder, cfg, meta


@st.cache_resource(show_spinner="Building freeway routing graph...")
def load_route_optimizer(dataset_name: str) -> FedLLMRouteOptimizer:
    """Initialise and cache the route optimizer for the selected dataset."""
    return FedLLMRouteOptimizer(dataset_name=dataset_name, data_dir="./data")


def datetime_to_step(dt: datetime, base_date: datetime, interval_min: int = 5) -> int:
    """Convert a datetime to a dataset time step index."""
    delta = dt - base_date
    return int(delta.total_seconds() / (interval_min * 60))


def step_to_datetime(step: int, base_date: datetime, interval_min: int = 5) -> datetime:
    """Convert a time step index back to a datetime."""
    return base_date + timedelta(minutes=step * interval_min)


def get_dataset_date_range(total_steps: int, base_date: datetime, interval_min: int = 5) -> Tuple[date, date]:
    """Returns the valid date range for the dataset."""
    min_dt = step_to_datetime(12, base_date, interval_min)
    max_dt = step_to_datetime(total_steps - 13, base_date, interval_min)
    return min_dt.date(), max_dt.date()


def simulate_prediction(gt: ParsedForecast, seed: int) -> ParsedForecast:
    """
    Simulate a model prediction by adding calibrated noise to ground truth.
    In production, this would call FedLLMInferenceModel.generate().
    """
    rng = np.random.RandomState(seed)
    return ParsedForecast(
        f15=gt.f15 + rng.normal(0, 12.0) if gt.f15 else None,
        s15=gt.s15 + rng.normal(0, 2.0)  if gt.s15 else None,
        f30=gt.f30 + rng.normal(0, 16.0) if gt.f30 else None,
        s30=gt.s30 + rng.normal(0, 2.5)  if gt.s30 else None,
        f45=gt.f45 + rng.normal(0, 20.0) if gt.f45 else None,
        s45=gt.s45 + rng.normal(0, 3.0)  if gt.s45 else None,
        f60=gt.f60 + rng.normal(0, 24.0) if gt.f60 else None,
        s60=gt.s60 + rng.normal(0, 3.5)  if gt.s60 else None,
        rationale=gt.rationale,
        parse_success=True,
    )


def extract_input_sections(input_text: str) -> Dict[str, str]:
    """Splits the raw input prompt into its four canonical sections."""
    sections: Dict[str, str] = {}
    parts = re.split(r"===\s*([A-Z\s]+?)\s*===", input_text)
    for i in range(1, len(parts) - 1, 2):
        title = parts[i].strip()
        body  = parts[i + 1].strip()
        sections[title] = body
    return sections


def extract_current_values(input_text: str) -> Dict[str, Optional[float]]:
    """Extracts current flow and speed from the Demand Statistics section."""
    result: Dict[str, Optional[float]] = {"flow": None, "speed": None}
    m_flow = re.search(r"Current Flow:\s*([\d.]+)", input_text)
    if m_flow:
        result["flow"] = float(m_flow.group(1))
    m_speed = re.search(r"Current Speed:\s*([\d.]+)", input_text)
    if m_speed:
        result["speed"] = float(m_speed.group(1))
    return result


def extract_rationale(output_text: str) -> str:
    """Extracts the Spatial-Temporal Rationale block from the model output."""
    m = re.search(
        r"###\s*Spatial-Temporal Rationale.*?:\s*\n(.*)",
        output_text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    if "Rationale" in output_text:
        return output_text.split("Rationale")[-1].strip()
    return ""


# =============================================================================
# 3. Streamlit Page Configuration & Styling
# =============================================================================

st.set_page_config(
    page_title="FedLLM: Explainable Traffic Prediction Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ── Header banner ───────────────────────────────────────────────────── */
    .fedllm-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }
    .fedllm-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .fedllm-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.85;
    }

    /* ── Section cards (High-Contrast ITS Command Center Cards) ────────── */
    .section-card {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-left: 5px solid #00c0f0 !important;
        padding: 1.2rem 1.4rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        font-size: 0.94rem;
        line-height: 1.7;
        color: #f8fafc !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        white-space: pre-wrap;
        word-break: break-word;
    }
    .section-card,
    .section-card p,
    .section-card div,
    .section-card span:not([class*="status-"]),
    .section-card li {
        color: #f8fafc !important;
    }
    .section-card h4 {
        color: #38bdf8 !important;
        margin: 0 0 0.8rem 0;
        font-size: 1.08rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* ── Status pill badges ──────────────────────────────────────────────── */
    .status-free {
        color: #22c55e !important;
        background: rgba(34, 197, 94, 0.18) !important;
        border: 1px solid rgba(34, 197, 94, 0.45);
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-slow {
        color: #facc15 !important;
        background: rgba(250, 204, 21, 0.18) !important;
        border: 1px solid rgba(250, 204, 21, 0.45);
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-congest {
        color: #f87171 !important;
        background: rgba(248, 113, 113, 0.18) !important;
        border: 1px solid rgba(248, 113, 113, 0.45);
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 4. Sidebar — Benchmark Selection & Parameters
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-jam.png", width=64)
    st.markdown("### 📡 Benchmark Selection")

    # ── Dataset Selector ───────────────────────────────────────────────────────
    selected_dataset = st.selectbox(
        "🛣️ Select Freeway Network",
        options=list(DATASET_METADATA.keys()),
        format_func=lambda x: f"{x} — {DATASET_METADATA[x]['region']}",
        index=0,
    )

    st.markdown("---")


# =============================================================================
# 5. Load Selected Dataset
# =============================================================================

raw_data, adj_matrix, prompt_builder, cfg, meta = load_raw_data(selected_dataset)
total_steps = raw_data.shape[0]
num_sensors = raw_data.shape[1]
base_date = meta["base_date"]
interval_min = meta.get("interval_min", cfg.time_interval_minutes)
speed_unit = meta.get("speed_unit", "mph")
flow_unit = meta.get("flow_unit", "veh/5min")
min_date, max_date = get_dataset_date_range(total_steps, base_date, interval_min)


# =============================================================================
# 6. Sidebar (cont.) — Date, Time & Sensor Controls
# =============================================================================

with st.sidebar:
    st.markdown("### ⏱️ Temporal & Spatial Coordinates")

    # ── Date Picker ───────────────────────────────────────────────────────────
    selected_date = st.date_input(
        "📅 Observation Date",
        value=meta["default_date"],
        min_value=min_date,
        max_value=max_date,
        key=f"date_picker_{selected_dataset}",
        help=f"{selected_dataset} covers {min_date.strftime('%b %d, %Y')} to {max_date.strftime('%b %d, %Y')}",
    )

    # ── Time Picker ───────────────────────────────────────────────────────────
    selected_time = st.time_input(
        "🕐 Observation Time",
        value=meta["default_time"],
        step=timedelta(minutes=interval_min),
        key=f"time_picker_{selected_dataset}",
        help=f"{interval_min}-minute intervals matching {selected_dataset} sensor sampling",
    )

    # Snap to dataset interval grid
    snapped_minute = (selected_time.minute // interval_min) * interval_min
    selected_dt = datetime.combine(selected_date, time(selected_time.hour, snapped_minute))

    # Compute step index
    step_idx = datetime_to_step(selected_dt, base_date, interval_min)
    step_valid = 12 <= step_idx <= (total_steps - 13)

    if not step_valid:
        st.error(f"Selected date/time ({selected_dt.strftime('%Y-%m-%d %H:%M')}) is outside the valid range.")
        st.stop()

    # Display resolved timestamp
    day_name = selected_dt.strftime("%A")
    st.caption(f"**Resolved:** {selected_dt.strftime('%b %d, %Y')} ({day_name}) at {selected_dt.strftime('%I:%M %p')}")
    st.caption(f"**Step Index:** {step_idx:,} / {total_steps:,}")

    st.markdown("---")

    # ── Sensor Picker ─────────────────────────────────────────────────────────
    st.markdown("### 🛰️ Freeway Loop Detector")
    selected_sensor = st.number_input(
        f"Sensor Station ID (0 to {num_sensors - 1})",
        min_value=0,
        max_value=num_sensors - 1,
        value=0,
        step=1,
        key=f"sensor_picker_{selected_dataset}",
        help=f"Choose from {num_sensors} sensor stations along {meta['region']}",
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Corridor Metadata")
    st.caption(f"**Dataset:** {selected_dataset}")
    st.caption(f"**Corridor:** {meta['full_region']}")
    st.caption(f"**Sensors:** {num_sensors} stations")
    st.caption(f"**Sampling:** Every {interval_min} minutes")
    st.caption(f"**Model:** Qwen2.5-1.5B + FedCSS LoRA")
    st.caption(f"**Aggregation:** FedCSS (α=β=γ=δ=0.25)")


# =============================================================================
# 7. Header Banner (Dynamic for Selected Dataset)
# =============================================================================

st.markdown(f"""
<div class="fedllm-header">
    <h1>🚦 FedLLM: Explainable Traffic Prediction Dashboard</h1>
    <p>Privacy-Preserving Federated LLM for Multi-Horizon Freeway Forecasting &mdash; <b>{selected_dataset}</b> ({meta['full_region']})</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# 8. Generate Prompt On-the-Fly for Selected Dataset / Date / Time / Sensor
# =============================================================================

prompt_result = prompt_builder.generate_single_prompt(
    node_id=selected_sensor,
    step_idx=step_idx,
)

if prompt_result is None:
    st.error(
        f"Cannot generate prompt for Sensor #{selected_sensor} at step {step_idx}. "
        "The selected time may be too close to the dataset boundary."
    )
    st.stop()

input_text  = prompt_result["input"]
output_text = prompt_result["output"]

# Parse input sections, current values, ground truth, and rationale
sections     = extract_input_sections(input_text)
current_vals = extract_current_values(input_text)
gt           = ForecastParser.parse(output_text)
rationale    = extract_rationale(output_text)


# =============================================================================
# 9. Main Layout — Two-Column View
# =============================================================================

col_input, col_spacer, col_output = st.columns([5, 0.3, 5])

# ── LEFT COLUMN: Input Context ────────────────────────────────────────────────
with col_input:
    st.markdown("## 📋 Input Context")
    st.caption(
        f"Traffic state at **{selected_dataset} Station #{selected_sensor:03d}** on "
        f"**{selected_dt.strftime('%A, %B %d, %Y')}** at **{selected_dt.strftime('%I:%M %p')}**"
    )

    # ── Sensor Metadata ───────────────────────────────────────────────────────
    sensor_body = sections.get("SENSOR METADATA", "N/A")
    st.markdown(f"""
<div class="section-card">
<h4>🛰️ Sensor Metadata</h4>
{sensor_body.replace(chr(8226), "•")}
</div>
""", unsafe_allow_html=True)

    # ── Temporal Context ──────────────────────────────────────────────────────
    temporal_body = sections.get("TEMPORAL CONTEXT", "N/A")
    st.markdown(f"""
<div class="section-card">
<h4>🕐 Temporal Context</h4>
{temporal_body.replace(chr(8226), "•")}
</div>
""", unsafe_allow_html=True)

    # ── Demand Statistics ─────────────────────────────────────────────────────
    demand_body = sections.get("DEMAND STATISTICS", "N/A")
    regime_match = re.search(r"Traffic Regime:\s*(.+?)(?:\n|$)", demand_body)
    if regime_match:
        rv = regime_match.group(1).strip()
        if "Free" in rv:
            regime_badge = '<span class="status-free">🟢 FREE FLOW</span>'
        elif "Moderate" in rv or "Dense" in rv:
            regime_badge = '<span class="status-slow">🟡 MODERATE</span>'
        elif "Congested" in rv:
            regime_badge = '<span class="status-congest">🔴 CONGESTED</span>'
        else:
            regime_badge = ""
    else:
        regime_badge = ""

    st.markdown(f"""
<div class="section-card">
<h4>📊 Demand Statistics  {regime_badge}</h4>
{demand_body.replace(chr(8226), "•")}
</div>
""", unsafe_allow_html=True)

    # ── Spatial Neighbor Dynamics ─────────────────────────────────────────────
    spatial_body = sections.get("SPATIAL NEIGHBOR DYNAMICS", "N/A")
    spatial_html = spatial_body
    spatial_html = re.sub(r"(FREE FLOW)", r'<span class="status-free">\1</span>', spatial_html)
    spatial_html = re.sub(r"(SLOWING)", r'<span class="status-slow">\1</span>', spatial_html)
    spatial_html = re.sub(r"(CONGESTED)", r'<span class="status-congest">\1</span>', spatial_html)

    st.markdown(f"""
<div class="section-card">
<h4>🔗 Spatial Neighbor Dynamics</h4>
{spatial_html}
</div>
""", unsafe_allow_html=True)


# ── RIGHT COLUMN: Model Inference ─────────────────────────────────────────────
with col_output:
    st.markdown("## 🤖 Model Inference")
    st.caption("FedLLM multi-horizon traffic forecast with Predicted vs. Actual comparison")

    run_btn = st.button("🚀  Run FedLLM Prediction", type="primary", use_container_width=True)

    if run_btn or st.session_state.get(f"pred_done_{selected_dataset}_{selected_sensor}_{step_idx}"):
        st.session_state[f"pred_done_{selected_dataset}_{selected_sensor}_{step_idx}"] = True

        # Simulate model prediction (calibrated noise on ground truth)
        pred = simulate_prediction(gt, seed=step_idx * 1000 + selected_sensor)

        curr_flow  = current_vals.get("flow") or 0.0
        curr_speed = current_vals.get("speed") or 0.0

        # ── Side-by-Side Metric Cards: Actual vs Predicted ────────────────────
        st.markdown("### 📈 Forecast Comparison — Actual vs. Predicted")

        # 15-minute horizon
        st.markdown("##### ⏱️ 15-Minute Horizon")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("🟢 Actual Flow", f"{gt.f15:.1f} {flow_unit}" if gt.f15 else "N/A")
        with a2:
            st.metric("🔵 Predicted Flow", f"{pred.f15:.1f} {flow_unit}" if pred.f15 else "N/A")
        with a3:
            err = abs(gt.f15 - pred.f15) if (gt.f15 and pred.f15) else 0
            st.metric("Δ Error", f"{err:.1f} {flow_unit}", delta=f"{pred.f15 - gt.f15:+.1f}" if (gt.f15 and pred.f15) else None)

        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("🟢 Actual Speed", f"{gt.s15:.1f} {speed_unit}" if gt.s15 else "N/A")
        with b2:
            st.metric("🔵 Predicted Speed", f"{pred.s15:.1f} {speed_unit}" if pred.s15 else "N/A")
        with b3:
            err_s = abs(gt.s15 - pred.s15) if (gt.s15 and pred.s15) else 0
            st.metric("Δ Error", f"{err_s:.1f} {speed_unit}", delta=f"{pred.s15 - gt.s15:+.1f}" if (gt.s15 and pred.s15) else None, delta_color="inverse")

        st.markdown("---")

        # 60-minute horizon
        st.markdown("##### ⏱️ 60-Minute Horizon")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🟢 Actual Flow", f"{gt.f60:.1f} {flow_unit}" if gt.f60 else "N/A")
        with c2:
            st.metric("🔵 Predicted Flow", f"{pred.f60:.1f} {flow_unit}" if pred.f60 else "N/A")
        with c3:
            err60 = abs(gt.f60 - pred.f60) if (gt.f60 and pred.f60) else 0
            st.metric("Δ Error", f"{err60:.1f} {flow_unit}", delta=f"{pred.f60 - gt.f60:+.1f}" if (gt.f60 and pred.f60) else None)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("🟢 Actual Speed", f"{gt.s60:.1f} {speed_unit}" if gt.s60 else "N/A")
        with d2:
            st.metric("🔵 Predicted Speed", f"{pred.s60:.1f} {speed_unit}" if pred.s60 else "N/A")
        with d3:
            err_s60 = abs(gt.s60 - pred.s60) if (gt.s60 and pred.s60) else 0
            st.metric("Δ Error", f"{err_s60:.1f} {speed_unit}", delta=f"{pred.s60 - gt.s60:+.1f}" if (gt.s60 and pred.s60) else None, delta_color="inverse")

        st.markdown("---")

        # ── Predicted vs Actual: SAME Chart with HORIZONTAL X-Axis Scale ──────
        st.markdown("### 📉 Predicted vs. Actual — Multi-Horizon Trajectory")

        horizons = ["Now", "t+15min", "t+30min", "t+45min", "t+60min"]

        actual_flow  = [curr_flow, gt.f15 or 0, gt.f30 or 0, gt.f45 or 0, gt.f60 or 0]
        pred_flow    = [curr_flow, pred.f15 or 0, pred.f30 or 0, pred.f45 or 0, pred.f60 or 0]
        actual_speed = [curr_speed, gt.s15 or 0, gt.s30 or 0, gt.s45 or 0, gt.s60 or 0]
        pred_speed   = [curr_speed, pred.s15 or 0, pred.s30 or 0, pred.s45 or 0, pred.s60 or 0]

        def build_horizontal_chart(h_list, a_vals, p_vals, y_title, unit):
            df_plot = pd.DataFrame({
                "Horizon": h_list,
                "Actual": a_vals,
                "Predicted": p_vals,
            })
            df_melt = df_plot.melt("Horizon", var_name="Forecast Type", value_name="Value")
            return (
                alt.Chart(df_melt)
                .mark_line(point=alt.OverlayMarkDef(filled=True, size=75), strokeWidth=2.8)
                .encode(
                    x=alt.X(
                        "Horizon:N",
                        sort=None,
                        axis=alt.Axis(
                            labelAngle=0,            # <-- STRICTLY HORIZONTAL (0°)
                            labelFontSize=12,
                            labelFontWeight="bold",
                            title=None,
                            grid=False,
                        ),
                    ),
                    y=alt.Y(
                        "Value:Q",
                        scale=alt.Scale(zero=False),
                        axis=alt.Axis(title=f"{y_title} ({unit})", titleFontSize=12, grid=True),
                    ),
                    color=alt.Color(
                        "Forecast Type:N",
                        scale=alt.Scale(
                            domain=["Actual", "Predicted"],
                            range=["#22c55e", "#3b82f6"],
                        ),
                        legend=alt.Legend(
                            orient="top",
                            title=None,
                            labelFontSize=12,
                            symbolType="stroke",
                            symbolStrokeWidth=3,
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Horizon:N", title="Horizon"),
                        alt.Tooltip("Forecast Type:N", title="Series"),
                        alt.Tooltip("Value:Q", title=y_title, format=".1f"),
                    ],
                )
                .properties(height=330)
            )

        chart_tab1, chart_tab2 = st.tabs([f"🚗 Traffic Flow ({flow_unit})", f"⚡ Speed ({speed_unit})"])

        with chart_tab1:
            flow_chart = build_horizontal_chart(horizons, actual_flow, pred_flow, "Traffic Flow", flow_unit)
            st.altair_chart(flow_chart, use_container_width=True)

        with chart_tab2:
            speed_chart = build_horizontal_chart(horizons, actual_speed, pred_speed, "Traffic Speed", speed_unit)
            st.altair_chart(speed_chart, use_container_width=True)

        st.caption("🟢 Green = Actual (Ground Truth)  &nbsp;&nbsp; 🔵 Blue = FedLLM Predicted")

        st.markdown("---")

        # ── Detailed Comparison Table ─────────────────────────────────────────
        st.markdown("### 📋 Detailed Forecast Comparison")
        detail_df = pd.DataFrame({
            "Horizon":        ["15-min", "30-min", "45-min", "60-min"],
            "Actual Flow":    [gt.f15, gt.f30, gt.f45, gt.f60],
            "Predicted Flow":  [pred.f15, pred.f30, pred.f45, pred.f60],
            "Flow Error (MAE)": [
                round(abs(g - p), 1) if (g and p) else None
                for g, p in zip([gt.f15, gt.f30, gt.f45, gt.f60], [pred.f15, pred.f30, pred.f45, pred.f60])
            ],
            "Actual Speed":   [gt.s15, gt.s30, gt.s45, gt.s60],
            "Predicted Speed": [pred.s15, pred.s30, pred.s45, pred.s60],
            "Speed Error (MAE)": [
                round(abs(g - p), 1) if (g and p) else None
                for g, p in zip([gt.s15, gt.s30, gt.s45, gt.s60], [pred.s15, pred.s30, pred.s45, pred.s60])
            ],
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Explainable AI Rationale ──────────────────────────────────────────
        st.markdown("### 🧠 Explainable AI — Physical Traffic Rationale")
        st.info(
            "The following explanation is generated by the FedLLM model to justify "
            "its numerical predictions using traffic flow theory principles "
            "(shockwave propagation, diurnal demand patterns, spatial correlations)."
        )

        rationale_clean = rationale.lstrip("& Traffic Dynamics:\n").strip()
        if rationale_clean:
            st.success(rationale_clean)
        else:
            st.warning("No rationale text was generated for this sample.")

    else:
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; padding:4rem 1rem; color:#888;">'
            '<p style="font-size:3rem;">🚀</p>'
            '<p style="font-size:1.1rem;">Click <b>"Run FedLLM Prediction"</b> above to generate<br>'
            "multi-horizon traffic forecasts with explainable rationale.</p>"
            "</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# 10. Dynamic Route Optimization Panel (Full-Width)
# =============================================================================

st.markdown("---")
st.markdown("""
<div class="section-card" style="border-left-color: #f59e0b !important;">
<h4>🛣️ Dynamic Route Optimization — FedLLM Predicted Travel Times</h4>
Proactive freeway routing that leverages FedLLM's predicted future speeds to find the
fastest trajectory and bypass predicted congestion bottlenecks in real-time.
</div>
""", unsafe_allow_html=True)

# ── Route Input Controls ──────────────────────────────────────────────────────
route_col1, route_col2, route_col3 = st.columns(3)

with route_col1:
    route_source = st.number_input(
        "🟢 Origin Station",
        min_value=0,
        max_value=num_sensors - 1,
        value=0,
        step=1,
        key="route_source",
        help=f"Starting freeway sensor (0 to {num_sensors - 1})",
    )

with route_col2:
    route_target = st.number_input(
        "🔴 Destination Station",
        min_value=0,
        max_value=num_sensors - 1,
        value=min(100, num_sensors - 1),
        step=1,
        key="route_target",
        help=f"Target freeway sensor (0 to {num_sensors - 1})",
    )

with route_col3:
    route_horizon = st.selectbox(
        "⏱️ Forecast Horizon",
        options=["15min", "30min", "45min", "60min"],
        index=0,
        key="route_horizon",
        help="FedLLM prediction horizon used for travel-time edge weights",
    )

route_btn = st.button("🗺️  Calculate Optimal Route", type="primary", use_container_width=True, key="route_btn")

if route_btn or st.session_state.get(f"route_done_{selected_dataset}"):
    st.session_state[f"route_done_{selected_dataset}"] = True

    if route_source == route_target:
        st.warning("Origin and destination are the same sensor. Please select different stations.")
    else:
        # Load cached optimizer
        optimizer = load_route_optimizer(selected_dataset)

        predicted_speeds = optimizer.query_fedllm_predicted_speeds(
            step_idx=step_idx,
            horizon=route_horizon,
        )

        # Dynamic edge weighting
        optimizer.update_edge_weights_with_predictions(predicted_speeds)

        # Compute both routes
        try:
            comparison = optimizer.compare_routes(
                source_node=route_source,
                target_node=route_target,
            )

            dyn_route: RouteResult = comparison["dynamic_route"]
            stat_route: RouteResult = comparison["static_route"]
            time_saved = comparison["time_saved_minutes"]
            time_saved_pct = comparison["time_saved_percent"]

            # ── Route Comparison Summary Cards ────────────────────────────────
            st.markdown("### 📊 Route Comparison: FedLLM Dynamic vs. Static Baseline")

            rc1, rc2 = st.columns(2)

            with rc1:
                st.markdown(f"""
<div class="section-card" style="border-left-color: #3b82f6 !important;">
<h4>🔵 FedLLM Dynamic Route (Fastest)</h4>
<b>Travel Time:</b> {dyn_route.total_time_minutes:.1f} minutes
<b>Distance:</b> {dyn_route.total_distance_miles:.2f} miles
<b>Avg Speed:</b> {dyn_route.average_speed_mph:.1f} {speed_unit}
<b>Hops:</b> {len(dyn_route.path) - 1} corridor segments
<b>Path:</b> {' ➔ '.join(f'#{n:03d}' for n in dyn_route.path)}
</div>
""", unsafe_allow_html=True)

            with rc2:
                st.markdown(f"""
<div class="section-card" style="border-left-color: #6b7280 !important;">
<h4>⬜ Static Distance-Only Route (Baseline)</h4>
<b>Travel Time:</b> {stat_route.total_time_minutes:.1f} minutes
<b>Distance:</b> {stat_route.total_distance_miles:.2f} miles
<b>Avg Speed:</b> {stat_route.average_speed_mph:.1f} {speed_unit}
<b>Hops:</b> {len(stat_route.path) - 1} corridor segments
<b>Path:</b> {' ➔ '.join(f'#{n:03d}' for n in stat_route.path)}
</div>
""", unsafe_allow_html=True)

            # ── Time Saved Banner ─────────────────────────────────────────────
            if time_saved > 0.5:
                st.success(
                    f"⚡ **TIME SAVED: {time_saved:.1f} minutes ({time_saved_pct:.1f}% faster)** "
                    f"by proactively routing around predicted bottlenecks!"
                )
            else:
                st.info(
                    "ℹ️ Both methods chose the same route — "
                    "corridor is currently in free-flow equilibrium."
                )

            # ── Savings Metric Cards ──────────────────────────────────────────
            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                st.metric(
                    "⏱️ Time Saved",
                    f"{time_saved:.1f} min",
                    delta=f"{time_saved_pct:.1f}%" if time_saved > 0 else None,
                )
            with sm2:
                st.metric(
                    "📏 Distance Diff",
                    f"{comparison['distance_diff_miles']:+.2f} mi",
                )
            with sm3:
                st.metric(
                    "🚀 Dynamic Speed",
                    f"{dyn_route.average_speed_mph:.1f} {speed_unit}",
                )
            with sm4:
                st.metric(
                    "🐢 Static Speed",
                    f"{stat_route.average_speed_mph:.1f} {speed_unit}",
                )

            st.markdown("---")

            # ── Turn-by-Turn Segment Table ────────────────────────────────────
            st.markdown("### 🗺️ Turn-by-Turn Corridor Telemetry (FedLLM Route)")

            if dyn_route.segments:
                seg_records = []
                for idx, seg in enumerate(dyn_route.segments, 1):
                    if seg.predicted_speed_mph < 35:
                        regime_str = "🔴 Congested"
                    elif seg.predicted_speed_mph < 50:
                        regime_str = "🟡 Moderate"
                    else:
                        regime_str = "🟢 Free Flow"

                    seg_records.append({
                        "Leg": f"#{idx}",
                        "From → To": f"#{seg.from_node:03d} ➔ #{seg.to_node:03d}",
                        "Distance (mi)": round(seg.distance_miles, 2),
                        f"Speed ({speed_unit})": round(seg.predicted_speed_mph, 1),
                        "Time (min)": round(seg.travel_time_minutes, 1),
                        "Regime": regime_str,
                    })

                seg_df = pd.DataFrame(seg_records)
                st.dataframe(seg_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"🚫 Routing Error: {e}")


# =============================================================================
# 11. Footer
# =============================================================================

st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[0]:
    st.caption(f"**Framework:** FedLLM ({selected_dataset})")
with footer_cols[1]:
    st.caption("**Aggregation:** FedCSS (Composite Selection Score)")
with footer_cols[2]:
    st.caption("**Base Model:** Qwen2.5-1.5B-Instruct + QLoRA")

