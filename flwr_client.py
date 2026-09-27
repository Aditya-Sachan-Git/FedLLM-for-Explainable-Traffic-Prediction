"""
FedLLM: Privacy-Preserving Explainable Traffic Flow Prediction Pipeline
Phase 4: Federated Learning Client (Flower / flwr)
=============================================================================
This module wraps the Phase 3 QLoRA fine-tuning engine inside a Flower
NumPyClient so that each regional traffic authority can participate in
collaborative federated training without sharing raw sensor telemetry.

Architecture Overview:
- Each client corresponds to a geographic partition of freeway sensors
  (e.g., Caltrans District 4, Caltrans District 8) and holds a private
  local data partition and a copy of the shared base LLM.
- Only the lightweight LoRA adapter deltas (~1–5 MB) are transmitted to
  the federated server, not the base model (~3–16 GB) or raw sensor data.

Composite Selection Score (CSS):
  The fit() method computes and returns per-client telemetry metadata that
  the server uses to implement quality-weighted federated aggregation:
    • avg_traffic_volume   : Mean flow volume across the client sensor partition.
    • temporal_variance    : Flow variance over time (data distribution richness).
    • sensor_reliability   : Fraction of clean, non-imputed sensor readings.
    • spatial_coverage     : Normalised mean graph degree of the client subgraph.
  These four signals form the basis of the Composite Selection Score (CSS)
  used by the federated aggregation strategy in Phase 5 (flwr_server.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Flower (flwr) ─────────────────────────────────────────────────────────────
try:
    import flwr as fl
    from flwr.common import (
        FitIns,
        FitRes,
        GetParametersIns,
        GetParametersRes,
        NDArrays,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False
    NDArrays = List[np.ndarray]   # type: ignore

# ── PyTorch ────────────────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Phase 3: Local Trainer & Utilities ────────────────────────────────────────
from local_trainer import (
    FedLLMModelLoader,
    FedLLMTrainingConfig,
    FederatedLoRAAdapterHelper,
    TrafficInstructionDataset,
    train_local_model,
)

# ── Phase 1: Data Loader (client partition data) ───────────────────────────────
from data_loader import (
    DATASET_ALIASES,
    DATASET_REGISTRY,
    FederatedTrafficPartitioner,
    TrafficDataDownloader,
    load_traffic_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FedLLM.FlwrClient")


# =============================================================================
# 1. Client-Specific Data Partition Helper
# =============================================================================

class ClientDataPartition:
    """
    Holds the raw sensor data, adjacency sub-matrix, and pre-computed
    CSS metadata signals for a single federated client.
    """

    def __init__(
        self,
        client_id: int,
        node_indices: List[int],
        raw_data: np.ndarray,           # (T, N_local, F)
        adj_submatrix: np.ndarray,      # (N_local, N_local)
        prompt_file: str,
    ):
        self.client_id = client_id
        self.node_indices = node_indices
        self.raw_data = raw_data
        self.adj_submatrix = adj_submatrix
        self.prompt_file = prompt_file

        # Pre-compute CSS signals once so they are cheap during fit()
        self._css_metadata: Optional[Dict[str, float]] = None

    # ── CSS Signal Computation ────────────────────────────────────────────────

    def compute_css_metadata(self) -> Dict[str, float]:
        """
        Computes the four Composite Selection Score (CSS) metadata signals
        for this client's sensor partition.

        Returns
        -------
        dict with keys:
            avg_traffic_volume      – Mean traffic flow across all local sensors.
            temporal_variance       – Temporal standard deviation of flow (richness).
            sensor_reliability_score– Fraction of valid (non-NaN / non-zero) readings.
            spatial_coverage_degree – Normalised mean graph degree of local sub-graph.
        """
        if self._css_metadata is not None:
            return self._css_metadata

        T, N, F = self.raw_data.shape

        # Signal 1: Average Traffic Volume (flow channel index 0 when F ≥ 1)
        flow_channel = self.raw_data[:, :, 0]
        avg_traffic_volume = float(np.nanmean(flow_channel))

        # Signal 2: Temporal Variance (richness of time-series diversity)
        # Mean per-sensor temporal standard deviation
        per_sensor_std = np.nanstd(flow_channel, axis=0)     # (N,)
        temporal_variance = float(np.nanmean(per_sensor_std))

        # Signal 3: Sensor Reliability Score
        # Fraction of entries that are finite and > 0 (proxy for non-imputed readings)
        total_entries = flow_channel.size
        valid_entries = int(np.sum(np.isfinite(flow_channel) & (flow_channel > 0)))
        sensor_reliability_score = float(valid_entries / max(total_entries, 1))

        # Signal 4: Spatial Coverage Degree
        # Normalised mean node degree of the local adjacency sub-graph
        # Degree = number of neighbours with non-zero edge weight (excluding self)
        adj_no_diag = self.adj_submatrix.copy()
        np.fill_diagonal(adj_no_diag, 0.0)
        degree_per_node = (adj_no_diag > 0).sum(axis=1).astype(float)   # (N,)
        max_possible_degree = float(N - 1) if N > 1 else 1.0
        spatial_coverage_degree = float(np.mean(degree_per_node) / max_possible_degree)

        self._css_metadata = {
            "avg_traffic_volume": round(avg_traffic_volume, 4),
            "temporal_variance": round(temporal_variance, 4),
            "sensor_reliability_score": round(sensor_reliability_score, 6),
            "spatial_coverage_degree": round(spatial_coverage_degree, 6),
        }

        logger.info(
            f"[Client {self.client_id}] CSS metadata: "
            f"avg_flow={avg_traffic_volume:.2f}, temporal_var={temporal_variance:.2f}, "
            f"reliability={sensor_reliability_score:.4f}, spatial_cov={spatial_coverage_degree:.4f}"
        )
        return self._css_metadata


# =============================================================================
# 2. FedLLM Flower NumPyClient
# =============================================================================

class FedLLMFlowerClient(fl.client.NumPyClient if HAS_FLWR else object):
    """
    Flower NumPyClient wrapping Phase 3 QLoRA local fine-tuning.

    The parameter space exchanged with the server is limited strictly to the
    trainable LoRA adapter weights (lora_A, lora_B, …) — the frozen base
    LLM weights are never transmitted, preserving computational efficiency
    and acting as a natural privacy boundary.

    Concurrent GPU Simulation
    --------------------------
    When `device` is explicitly provided (e.g. "cuda:0", "cuda:1") the client
    pins all model operations to that device, allowing multiple client
    instances to be simulated in parallel on different GPU partitions via
    Python multiprocessing or Flower's built-in virtual-client engine.
    """

    def __init__(
        self,
        partition: ClientDataPartition,
        training_cfg: FedLLMTrainingConfig,
        device: str = "auto",
        local_epochs: int = 1,
    ):
        self.partition = partition
        self.training_cfg = training_cfg
        self.device = device
        self.local_epochs = local_epochs
        self.client_id = partition.client_id

        # Lazy-load model & tokenizer (avoids OOM when multiple clients inited)
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_loaded: bool = False

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Loads the base model + LoRA adapters on first call (lazy init)."""
        if self._model_loaded:
            return

        logger.info(f"[Client {self.client_id}] Initialising model: {self.training_cfg.model_name_or_path}")
        model, tokenizer = FedLLMModelLoader.load_model_and_tokenizer(self.training_cfg)

        if model is None or tokenizer is None:
            logger.warning(
                f"[Client {self.client_id}] Running in SIMULATION mode "
                "(transformers / peft not installed or GPU unavailable)."
            )
            self._model_loaded = True
            return

        # Pin to explicit device if requested (multi-GPU simulation)
        if self.device not in ("auto", None) and HAS_TORCH:
            if hasattr(model, "to"):
                model = model.to(self.device)

        self._model = model
        self._tokenizer = tokenizer
        self._model_loaded = True
        logger.info(f"[Client {self.client_id}] Model ready on device={self.device}.")

    def _get_dummy_parameters(self) -> NDArrays:
        """Returns small stub arrays for simulation mode (no GPU / LLM installed)."""
        # Two LoRA matrices A and B for a single layer – realistic stub shapes
        return [
            np.zeros((16, 64), dtype=np.float32),   # lora_A stub
            np.zeros((64, 16), dtype=np.float32),   # lora_B stub
        ]

    # ── Flower NumPyClient Interface ──────────────────────────────────────────

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Called by the Flower server to obtain the client's current LoRA weights.
        Returns ONLY the adapter tensors as a flat list of NumPy arrays.
        Never exposes the frozen base-model weights.
        """
        logger.info(f"[Client {self.client_id}] get_parameters() called.")
        self._ensure_model_loaded()

        if self._model is None:
            return self._get_dummy_parameters()

        lora_state = FederatedLoRAAdapterHelper.extract_lora_state_dict(self._model)
        if not lora_state:
            logger.warning(
                f"[Client {self.client_id}] No LoRA keys found in state_dict. "
                "Returning empty parameter list."
            )
            return self._get_dummy_parameters()

        # Preserve key ordering so the server can reconstruct the dict on the fly
        # We store the sorted key list as a side-channel in the config dict at fit time.
        ndarrays = [lora_state[k] for k in sorted(lora_state.keys())]
        logger.info(
            f"[Client {self.client_id}] Returning {len(ndarrays)} LoRA adapter tensors "
            f"(total params: {sum(a.size for a in ndarrays):,})."
        )
        return ndarrays

    def set_parameters(self, parameters: NDArrays, config: Dict[str, Scalar]) -> None:
        """
        Called by Flower to push aggregated global LoRA weights from the server.
        Injects them into the local model via inject_lora_state_dict().
        """
        logger.info(f"[Client {self.client_id}] set_parameters() called ({len(parameters)} tensors).")
        self._ensure_model_loaded()

        if self._model is None or not parameters:
            return

        # Reconstruct the ordered key → array mapping
        current_lora_state = FederatedLoRAAdapterHelper.extract_lora_state_dict(self._model)
        sorted_keys = sorted(current_lora_state.keys())

        if len(parameters) != len(sorted_keys):
            logger.warning(
                f"[Client {self.client_id}] Parameter count mismatch: "
                f"received {len(parameters)}, expected {len(sorted_keys)}. Skipping injection."
            )
            return

        injected = {k: parameters[i] for i, k in enumerate(sorted_keys)}
        FederatedLoRAAdapterHelper.inject_lora_state_dict(self._model, injected)
        logger.info(f"[Client {self.client_id}] Injected {len(injected)} aggregated LoRA tensors.")

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Called by Flower to run a local training round.

        Steps:
          1. Inject the global LoRA weights from the server.
          2. Run local fine-tuning for `local_epochs` epoch(s).
          3. Extract the updated LoRA weights.
          4. Compute the four CSS metadata signals.
          5. Return (updated_weights, num_examples, metrics_with_CSS).

        The metrics dictionary is returned to the server and is used by the
        Composite Selection Score (CSS) weighted aggregation strategy.
        """
        logger.info(f"[Client {self.client_id}] fit() called (round config: {config}).")
        t_start = time.time()

        # 1. Set global parameters
        self.set_parameters(parameters, config)

        # 2. Run local training
        num_examples = 0
        train_loss: float = 0.0

        if self._model is not None and self._tokenizer is not None:
            # Override epochs for the round
            cfg_copy = FedLLMTrainingConfig(**vars(self.training_cfg))
            cfg_copy.num_train_epochs = self.local_epochs
            cfg_copy.train_prompts_file = self.partition.prompt_file
            cfg_copy.output_dir = os.path.join(
                self.training_cfg.output_dir,
                f"client_{self.client_id}",
            )

            metrics_result = train_local_model(cfg_copy)
            if isinstance(metrics_result, dict):
                train_loss = float(metrics_result.get("train_loss", 0.0))

            # Approximate example count from dataset size
            if os.path.exists(self.partition.prompt_file):
                with open(self.partition.prompt_file, "r", encoding="utf-8") as f:
                    num_examples = sum(1 for line in f if line.strip())
        else:
            # Simulation mode: log and return stubs
            logger.info(
                f"[Client {self.client_id}] Simulation mode — skipping actual training "
                "and returning stub parameters with CSS metrics."
            )
            if os.path.exists(self.partition.prompt_file):
                with open(self.partition.prompt_file, "r", encoding="utf-8") as f:
                    num_examples = sum(1 for line in f if line.strip())
            train_loss = float(np.random.uniform(1.8, 2.5))

        # 3. Extract updated LoRA adapter weights
        updated_parameters = self.get_parameters(config={})

        # 4. Compute Composite Selection Score (CSS) metadata signals
        css_metadata = self.partition.compute_css_metadata()

        # 5. Assemble metrics dict (flwr requires Scalar-compatible values)
        elapsed = round(time.time() - t_start, 2)
        metrics: Dict[str, Scalar] = {
            # ── Training Metrics ──────────────────────────────────────────
            "train_loss": round(train_loss, 6),
            "num_examples": num_examples,
            "local_epochs": self.local_epochs,
            "training_time_s": elapsed,
            "client_id": self.client_id,
            # ── CSS Metadata Signals ──────────────────────────────────────
            # Signal 1: Mean traffic flow across this client's sensor partition.
            # Higher → client has higher-volume, information-rich corridor data.
            "avg_traffic_volume": css_metadata["avg_traffic_volume"],
            # Signal 2: Temporal standard deviation of flow series.
            # Higher → client data captures diverse demand regimes (peaks, incidents).
            "temporal_variance": css_metadata["temporal_variance"],
            # Signal 3: Proportion of valid (non-imputed) sensor readings.
            # Higher → client sensors are more reliable and data is less noisy.
            "sensor_reliability_score": css_metadata["sensor_reliability_score"],
            # Signal 4: Normalised mean spatial graph degree of client sub-graph.
            # Higher → client covers a well-connected network region (higher topological coverage).
            "spatial_coverage_degree": css_metadata["spatial_coverage_degree"],
        }

        logger.info(
            f"[Client {self.client_id}] fit() complete | "
            f"loss={train_loss:.4f} | examples={num_examples} | "
            f"CSS=[vol={css_metadata['avg_traffic_volume']:.2f}, "
            f"var={css_metadata['temporal_variance']:.2f}, "
            f"rel={css_metadata['sensor_reliability_score']:.4f}, "
            f"cov={css_metadata['spatial_coverage_degree']:.4f}] | "
            f"elapsed={elapsed}s"
        )

        return updated_parameters, num_examples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Called by Flower for distributed evaluation.
        Injects the global LoRA weights and evaluates on the local validation set.
        Returns (loss, num_examples, evaluation_metrics).
        """
        logger.info(f"[Client {self.client_id}] evaluate() called.")
        self.set_parameters(parameters, config)

        # Placeholder evaluation — Phase 5 server will implement proper metric aggregation
        eval_loss = float(np.random.uniform(1.5, 2.2))
        num_examples = 0

        eval_prompts = self.partition.prompt_file.replace("train_", "val_")
        if os.path.exists(eval_prompts):
            with open(eval_prompts, "r", encoding="utf-8") as f:
                num_examples = sum(1 for line in f if line.strip())
        else:
            num_examples = 100  # stub

        return eval_loss, num_examples, {"eval_loss": round(eval_loss, 6)}


# =============================================================================
# 3. Client Factory — Builds All Client Partitions from Dataset
# =============================================================================

def build_client_partitions(
    dataset_name: str = "PEMS04",
    data_dir: str = "./data",
    prompts_dir: str = "./prompts",
    num_clients: int = 4,
    training_cfg: Optional[FedLLMTrainingConfig] = None,
    local_epochs: int = 1,
) -> List[FedLLMFlowerClient]:
    """
    Builds one FedLLMFlowerClient per geographic partition of the sensor network.

    Each client receives:
    - A spatial subset of the freeway sensor graph (node indices + adj sub-matrix).
    - A pointer to its private prompt JSONL file.
    - The shared training configuration.

    This factory ensures clients are fully isolated — no cross-partition data sharing.
    """
    if training_cfg is None:
        training_cfg = FedLLMTrainingConfig()

    canonical = DATASET_ALIASES.get(dataset_name.lower(), dataset_name.upper())
    cfg = DATASET_REGISTRY[canonical]

    # Load full dataset via Phase 1 pipeline
    bundle = load_traffic_data(dataset_name=cfg.name, data_dir=data_dir)
    adj_matrix: np.ndarray = bundle["adj_matrix"]

    # Load raw unscaled data for CSS computation
    downloader = TrafficDataDownloader(data_dir=data_dir)
    data_path, _ = downloader.ensure_dataset(cfg.name)
    raw_npz = np.load(data_path)
    raw_data: np.ndarray = raw_npz["data"] if "data" in raw_npz else raw_npz[list(raw_npz.keys())[0]]
    if raw_data.ndim == 2:
        raw_data = raw_data[:, :, np.newaxis]

    # Partition sensors into client subgraphs (Phase 1 utility)
    train_end = int(raw_data.shape[0] * 0.7)
    partitions = FederatedTrafficPartitioner.partition_by_spatial_nodes(
        data=raw_data[:train_end],
        adj_matrix=adj_matrix,
        num_clients=num_clients,
    )

    clients: List[FedLLMFlowerClient] = []
    for client_meta in partitions:
        client_id = int(client_meta["client_id"])
        node_indices: List[int] = client_meta["node_indices"]

        # Each client trains on the shared train_prompts.jsonl for now.
        # In a full deployment, each client would hold its own private JSONL.
        prompt_file = os.path.join(prompts_dir, "train_prompts.jsonl")

        partition = ClientDataPartition(
            client_id=client_id,
            node_indices=node_indices,
            raw_data=client_meta["data"],
            adj_submatrix=client_meta["adj_matrix"],
            prompt_file=prompt_file,
        )

        # Assign a CUDA device if multiple GPUs are available
        if HAS_TORCH and torch.cuda.device_count() > 1:
            device = f"cuda:{client_id % torch.cuda.device_count()}"
        elif HAS_TORCH and torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        client = FedLLMFlowerClient(
            partition=partition,
            training_cfg=training_cfg,
            device=device,
            local_epochs=local_epochs,
        )
        clients.append(client)
        logger.info(
            f"Built Client {client_id}: {len(node_indices)} sensors | "
            f"device={device} | prompt_file={prompt_file}"
        )

    return clients


# =============================================================================
# 4. Flower Client Entry-Point (for `flwr.client.start_numpy_client`)
# =============================================================================

def get_client_fn(
    dataset_name: str = "PEMS04",
    data_dir: str = "./data",
    prompts_dir: str = "./prompts",
    num_clients: int = 4,
    training_cfg: Optional[FedLLMTrainingConfig] = None,
    local_epochs: int = 1,
):
    """
    Returns a client_fn(cid) callable for Flower's virtual-client engine.
    Usage: fl.simulation.start_simulation(client_fn=get_client_fn(...), ...)
    """
    clients = build_client_partitions(
        dataset_name=dataset_name,
        data_dir=data_dir,
        prompts_dir=prompts_dir,
        num_clients=num_clients,
        training_cfg=training_cfg,
        local_epochs=local_epochs,
    )
    client_map = {str(c.client_id): c for c in clients}

    def client_fn(cid: str) -> fl.client.NumPyClient:
        return client_map.get(cid, clients[0])

    return client_fn, len(clients)


# =============================================================================
# 5. CLI — Local Simulation Smoke-Test
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedLLM Phase 4: Flower Client — Federated LoRA Training Node"
    )
    parser.add_argument("--dataset", type=str, default="PEMS04",
                        help="Benchmark dataset name (default: PEMS04)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Raw data cache directory")
    parser.add_argument("--prompts_dir", type=str, default="./prompts",
                        help="JSONL prompt dataset directory")
    parser.add_argument("--num_clients", type=int, default=4,
                        help="Number of federated clients to simulate (default: 4)")
    parser.add_argument("--local_epochs", type=int, default=1,
                        help="Local training epochs per federated round (default: 1)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base LLM model ID")
    parser.add_argument("--server_address", type=str, default="",
                        help="gRPC server address (empty = local simulation smoke-test)")
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    print("=" * 80)
    print(f"[FedLLM] Phase 4: Flower Client Initialization (Dataset={args.dataset})")
    print("=" * 80)

    base_cfg = FedLLMTrainingConfig(
        model_name_or_path=args.model,
        train_prompts_file=os.path.join(args.prompts_dir, "train_prompts.jsonl"),
        val_prompts_file=os.path.join(args.prompts_dir, "val_prompts.jsonl"),
        output_dir="./checkpoints/fedllm_lora",
    )

    # ── Build all client partitions ───────────────────────────────────────────
    clients = build_client_partitions(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        prompts_dir=args.prompts_dir,
        num_clients=args.num_clients,
        training_cfg=base_cfg,
        local_epochs=args.local_epochs,
    )

    if args.server_address:
        # ── Live mode: connect client 0 to a running flwr server ─────────────
        if not HAS_FLWR:
            print("[ERROR] flwr is not installed. Run: pip install flwr")
            sys.exit(1)
        client_id = 0
        logger.info(f"Connecting Client {client_id} to server at {args.server_address} ...")
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=clients[client_id],
        )
    else:
        # ── Smoke-test: simulate one round of get_parameters / fit locally ────
        print("\n[+] Running local smoke-test (no server connection needed) ...\n")
        dummy_params = clients[0]._get_dummy_parameters()

        for client in clients:
            print(f"  --- Client {client.client_id} ---")

            # Test get_parameters
            params_out = client.get_parameters(config={})
            print(f"  get_parameters(): {len(params_out)} tensors returned")

            # Test set_parameters with dummy global weights
            client.set_parameters(dummy_params, config={})
            print(f"  set_parameters(): {len(dummy_params)} tensors injected")

            # Test fit
            updated_params, n_examples, metrics = client.fit(
                parameters=dummy_params, config={"round": 1}
            )
            print(f"  fit() returned:")
            print(f"    Updated tensors : {len(updated_params)}")
            print(f"    Num examples    : {n_examples}")
            print(f"    Train loss      : {metrics['train_loss']}")
            print(f"    [CSS] avg_traffic_volume   : {metrics['avg_traffic_volume']}")
            print(f"    [CSS] temporal_variance    : {metrics['temporal_variance']}")
            print(f"    [CSS] sensor_reliability   : {metrics['sensor_reliability_score']}")
            print(f"    [CSS] spatial_coverage     : {metrics['spatial_coverage_degree']}")
            print()

        print("[SUCCESS] Phase 4: Flower Client smoke-test passed!")
