"""
FedLLM: Privacy-Preserving Explainable Traffic Flow Prediction Pipeline
Phase 5: Federated Learning Server — FedCSS Custom Aggregation Strategy
=============================================================================
This module implements the central Flower federated server for FedLLM. The
core novelty is the FedCSS (Federated Composite Selection Score) aggregation
strategy, which replaces standard FedAvg sample-count weighting with a
quality-aware, data-heterogeneity-aware weighted aggregation of LoRA adapters.

Why Standard FedAvg Fails Here
--------------------------------
FedAvg weights each client's parameter update proportionally to the number of
local training examples:

    w_i = n_i / Σ_j n_j

This creates three failure modes for non-IID traffic federated learning:

  1. Volume Bias: Clients covering high-flow urban freeways contribute
     disproportionately, causing the global model to ignore low-volume
     but safety-critical rural or arterial segments.

  2. Temporal Homogeneity Collapse: A client whose sensors all peak at
     rush-hour dominates the temporal distribution of the global adapter,
     degrading generalisation to overnight / weekend conditions.

  3. Structural Drift: Clients with noisy, highly-imputed sensor readings
     inject gradient noise at the same weight as clients with clean data.

The FedCSS Solution
--------------------
FedCSS replaces the scalar n_i with a multi-dimensional Composite Selection
Score that jointly captures four orthogonal data quality dimensions:

    CSS_i = (α · V̂_i) + (β · T̂_i) + (γ · R_i) + (δ · C_i)

Where (all terms are min-max normalised across the active client set):

    V̂_i  — avg_traffic_volume    (normalised mean flow)
    T̂_i  — temporal_variance     (normalised σ of flow series)
    R_i  — sensor_reliability_score  (fraction of valid readings ∈ [0,1])
    C_i  — spatial_coverage_degree   (normalised mean graph degree ∈ [0,1])

    α, β, γ, δ  — configurable importance weights (default: 0.25 each).

The LoRA-specific aggregation formula for layer tensor l is:

    Δθ_l^* = Σ_i  (CSS_i / Σ_j CSS_j)  ·  Δθ_l^(i)

This replaces naive FedAvg so that:
  - Structurally diverse (high spatial_coverage) corridors contribute more.
  - Temporally rich (high temporal_variance) data shifts temporal adaptation.
  - Reliable sensors (high reliability) dominate over imputed data.
  - High-volume corridors (high traffic_volume) receive deserved weight.

After each round the aggregated global adapter is saved to disk in NumPy
`.npy` format (ready for conversion to safetensors in Phase 6).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ── Flower ─────────────────────────────────────────────────────────────────────
try:
    import flwr as fl
    from flwr.common import (
        EvaluateIns,
        EvaluateRes,
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
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg, Strategy
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False
    # Provide stub types so the module is importable for diagnostics / offline use
    NDArrays = List[np.ndarray]      # type: ignore
    Scalar = Union[int, float, str, bool, bytes]   # type: ignore
    ClientProxy = Any                # type: ignore

# ── Optional safetensors export ────────────────────────────────────────────────
try:
    from safetensors.numpy import save_file as safetensors_save
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FedLLM.FlwrServer")


# =============================================================================
# 1. CSS Weight Calculator
# =============================================================================

class CSSWeightCalculator:
    """
    Computes Composite Selection Score (CSS) weights for all active clients
    in a single federated round.

    CSS_i = α·V̂_i  +  β·T̂_i  +  γ·R_i  +  δ·C_i

    Where  ^  denotes min-max normalisation across the active client set so
    that all four signals live on [0, 1] before weighting.

    The final aggregation weight for client i is:

        w_i = CSS_i / Σ_j CSS_j

    This ensures the weights sum to 1 (a proper convex combination).
    """

    def __init__(
        self,
        alpha: float = 0.25,   # weight for avg_traffic_volume
        beta:  float = 0.25,   # weight for temporal_variance
        gamma: float = 0.25,   # weight for sensor_reliability_score
        delta: float = 0.25,   # weight for spatial_coverage_degree
        epsilon: float = 1e-8, # numerical stability floor
    ):
        assert abs(alpha + beta + gamma + delta - 1.0) < 1e-6, (
            "CSS importance weights (α, β, γ, δ) must sum to 1.0."
        )
        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma
        self.delta   = delta
        self.epsilon = epsilon

    @staticmethod
    def _min_max_normalise(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Min-max normalises a 1-D array to [0, 1].
        If all values are identical (zero range), returns a uniform 1/N vector.
        """
        v_min, v_max = values.min(), values.max()
        rng = v_max - v_min
        if rng < eps:
            # All clients have the same signal value → equal contribution
            return np.ones_like(values) / len(values)
        return (values - v_min) / rng

    def compute(
        self,
        client_metrics: List[Dict[str, Scalar]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CSS weights for a list of client metric dictionaries.

        Parameters
        ----------
        client_metrics : list of per-client metric dicts (from fit() results).
            Each dict must contain:
                'avg_traffic_volume', 'temporal_variance',
                'sensor_reliability_score', 'spatial_coverage_degree'

        Returns
        -------
        css_scores  : np.ndarray of shape (K,) — raw CSS values before L1-norm.
        css_weights : np.ndarray of shape (K,) — normalised aggregation weights
                      summing to 1.0.
        """
        K = len(client_metrics)

        # ── Extract the four raw signal vectors ──────────────────────────────
        V = np.array([float(m.get("avg_traffic_volume",    0.0)) for m in client_metrics])
        T = np.array([float(m.get("temporal_variance",     0.0)) for m in client_metrics])
        R = np.array([float(m.get("sensor_reliability_score", 1.0)) for m in client_metrics])
        C = np.array([float(m.get("spatial_coverage_degree",  0.0)) for m in client_metrics])

        # ── Min-max normalise each signal independently ───────────────────────
        # Normalisation is done across the active client set in this round so
        # that a single outlier cannot distort the global scale.
        V_hat = self._min_max_normalise(V, self.epsilon)
        T_hat = self._min_max_normalise(T, self.epsilon)
        R_hat = self._min_max_normalise(R, self.epsilon)   # already ∈ [0,1]
        C_hat = self._min_max_normalise(C, self.epsilon)   # already ∈ [0,1]

        # ── CSS composite score ──────────────────────────────────────────────
        #   CSS_i = α·V̂_i  +  β·T̂_i  +  γ·R̂_i  +  δ·Ĉ_i
        css_scores = (
            self.alpha * V_hat +
            self.beta  * T_hat +
            self.gamma * R_hat +
            self.delta * C_hat
        )

        # ── L1 normalise → aggregation weights summing to 1 ──────────────────
        total = css_scores.sum()
        if total < self.epsilon:
            # Degenerate case (all zeros) → fall back to uniform weighting
            logger.warning("All CSS scores are near-zero; falling back to uniform weighting.")
            css_weights = np.ones(K, dtype=np.float64) / K
        else:
            css_weights = css_scores / total

        return css_scores, css_weights


# =============================================================================
# 2. CSS-Weighted LoRA Aggregation
# =============================================================================

def css_weighted_aggregate(
    results: List[Tuple[NDArrays, int, Dict[str, Scalar]]],
    css_weights: np.ndarray,
) -> NDArrays:
    """
    Performs CSS-weighted aggregation of LoRA parameter tensors.

    Standard FedAvg aggregation (for reference):
        Δθ_l^* = Σ_i (n_i / Σ_j n_j) · Δθ_l^(i)

    FedCSS aggregation (our novelty):
        Δθ_l^* = Σ_i  w_i  ·  Δθ_l^(i)
        where w_i = CSS_i / Σ_j CSS_j  (see CSSWeightCalculator.compute())

    This ensures that the global adapter reflects data quality and structural
    diversity rather than just raw sample volume.

    Parameters
    ----------
    results     : List of (ndarrays, num_examples, metrics) tuples — one per
                  client participating in this round.
    css_weights : np.ndarray of shape (K,) summing to 1.0 — CSS-derived
                  aggregation weights computed by CSSWeightCalculator.

    Returns
    -------
    aggregated : NDArrays — list of aggregated LoRA tensors in the same layer
                 order as the client parameter lists.
    """
    assert len(results) == len(css_weights), (
        f"Mismatch: {len(results)} client results vs {len(css_weights)} CSS weights."
    )

    # Number of parameter tensors (= number of LoRA A/B matrices across layers)
    num_tensors = len(results[0][0])

    aggregated: NDArrays = []
    for tensor_idx in range(num_tensors):
        # Weighted sum:  w_i · Δθ_l^(i)  across all clients
        weighted_tensor = np.zeros_like(
            results[0][0][tensor_idx], dtype=np.float64
        )
        for client_idx, (ndarrays, _num_examples, _metrics) in enumerate(results):
            client_tensor = ndarrays[tensor_idx].astype(np.float64)
            weighted_tensor += css_weights[client_idx] * client_tensor

        aggregated.append(weighted_tensor.astype(np.float32))

    return aggregated


# =============================================================================
# 3. FedCSS — Custom Flower Strategy
# =============================================================================

class FedCSS(FedAvg if HAS_FLWR else object):
    """
    FedCSS: Federated Composite Selection Score Aggregation Strategy.

    Inherits from flwr.server.strategy.FedAvg and overrides aggregate_fit()
    to replace sample-count weighting with CSS-quality weighting.

    All other FedAvg behaviours (client sampling, evaluation scheduling,
    initial parameter handling) are preserved.

    Mathematical Derivation of CSS
    --------------------------------
    For a round with K participating clients, let:

        m_i ∈ ℝ⁴  =  [V_i, T_i, R_i, C_i]     (client i's CSS signal vector)

    Step 1 — Normalise each dimension across clients:
        V̂_i = (V_i − min_j V_j) / (max_j V_j − min_j V_j + ε)
        T̂_i = (T_i − min_j T_j) / (max_j T_j − min_j T_j + ε)
        R̂_i = (R_i − min_j R_j) / (max_j R_j − min_j R_j + ε)
        Ĉ_i = (C_i − min_j C_j) / (max_j C_j − min_j C_j + ε)

    Step 2 — Compute the composite score:
        CSS_i = α·V̂_i + β·T̂_i + γ·R̂_i + δ·Ĉ_i
        where α + β + γ + δ = 1.

    Step 3 — L1-normalise to get convex-combination weights:
        w_i = CSS_i / Σ_j CSS_j,   Σ_i w_i = 1

    Step 4 — Aggregate LoRA tensor l:
        Δθ_l^* = Σ_i w_i · Δθ_l^(i)

    Checkpoint Saving
    -----------------
    After each successful aggregate_fit() call the global LoRA adapter is
    saved to `<checkpoint_dir>/round_<R>/` in two formats:
      • adapter_<key>.npy  — raw NumPy arrays for inspection / analysis.
      • adapter.safetensors — HuggingFace-compatible format (if safetensors
                              is installed) for direct PEFT model loading.
    """

    def __init__(
        self,
        # CSS importance weights (must sum to 1.0)
        css_alpha: float = 0.25,
        css_beta:  float = 0.25,
        css_gamma: float = 0.25,
        css_delta: float = 0.25,
        # Checkpoint directory
        checkpoint_dir: str = "./checkpoints/global_adapter",
        # Pass remaining kwargs to FedAvg (e.g. fraction_fit, min_available_clients)
        **fedavg_kwargs: Any,
    ):
        if HAS_FLWR:
            super().__init__(**fedavg_kwargs)

        self.css_calculator = CSSWeightCalculator(
            alpha=css_alpha,
            beta=css_beta,
            gamma=css_gamma,
            delta=css_delta,
        )
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Ordered adapter key list — populated on the first successful round.
        # Clients sort their LoRA keys alphabetically before stacking into NDArrays,
        # so the server can reconstruct the key→array mapping.
        self._adapter_keys: Optional[List[str]] = None

        # Per-round history for downstream analysis
        self.round_history: List[Dict[str, Any]] = []

        logger.info(
            f"FedCSS strategy initialised | "
            f"CSS weights: α={css_alpha}, β={css_beta}, γ={css_gamma}, δ={css_delta} | "
            f"Checkpoint dir: {checkpoint_dir}"
        )

    # ── Helper ────────────────────────────────────────────────────────────────

    def _save_global_adapter(
        self,
        aggregated: NDArrays,
        round_num: int,
    ) -> str:
        """
        Persists the aggregated global LoRA adapter after each federated round.

        Saves:
          1. One `.npy` file per adapter tensor (always, no extra dependencies).
          2. One `adapter.safetensors` file (if safetensors package is available).
          3. A `round_metadata.json` sidecar with CSS diagnostics.

        Returns the path to the round checkpoint directory.
        """
        round_dir = os.path.join(self.checkpoint_dir, f"round_{round_num:04d}")
        os.makedirs(round_dir, exist_ok=True)

        # ── Build a key→array mapping ─────────────────────────────────────────
        # If we have recorded adapter key names (from client metadata), use them.
        # Otherwise fall back to integer indices.
        if self._adapter_keys and len(self._adapter_keys) == len(aggregated):
            key_array_map = {k: arr for k, arr in zip(self._adapter_keys, aggregated)}
        else:
            key_array_map = {f"adapter_layer_{i:04d}": arr for i, arr in enumerate(aggregated)}

        # ── Save as .npy (always available) ───────────────────────────────────
        for key, arr in key_array_map.items():
            # Replace path separators in keys (e.g. "base_model.model.layers.0...")
            safe_key = key.replace(".", "_").replace("/", "_")
            np.save(os.path.join(round_dir, f"{safe_key}.npy"), arr)

        # ── Save as .safetensors (optional, HuggingFace-compatible) ───────────
        if HAS_SAFETENSORS:
            safetensors_path = os.path.join(round_dir, "adapter.safetensors")
            safetensors_save(key_array_map, safetensors_path)
            logger.info(f"  Saved adapter.safetensors → {safetensors_path}")
        else:
            logger.info(
                "  safetensors package not installed; skipping .safetensors export. "
                "Install via: pip install safetensors"
            )

        logger.info(
            f"[Round {round_num}] Global adapter saved: {len(aggregated)} tensors "
            f"→ {round_dir}"
        )
        return round_dir

    # ── Core FedAvg Override: aggregate_fit ───────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        CSS-weighted aggregation of LoRA adapter tensors.

        This method is called by the Flower server at the end of each round
        after all selected clients have returned their updated LoRA weights.

        Override of FedAvg.aggregate_fit() — replaces sample-count weighting
        with CSS quality-weighting (see class docstring for full mathematics).

        Parameters
        ----------
        server_round : Current federated round number (1-indexed).
        results      : List of (ClientProxy, FitRes) pairs — one per client
                       that successfully completed a local training round.
        failures     : List of clients or exceptions that failed this round.

        Returns
        -------
        parameters   : Flower Parameters object wrapping the aggregated adapter.
        metrics      : Round-level aggregated metrics dict for Flower history.
        """
        t0 = time.time()

        # ── Guard: nothing to aggregate ───────────────────────────────────────
        if not results:
            logger.warning(f"[Round {server_round}] No successful client results to aggregate.")
            return None, {}

        # ── Log any failures ──────────────────────────────────────────────────
        if failures:
            logger.warning(
                f"[Round {server_round}] {len(failures)} client(s) failed this round "
                f"and will be excluded from aggregation."
            )

        # ── Step 1: Unpack client results ─────────────────────────────────────
        # Each result is a (ClientProxy, FitRes) tuple.
        # FitRes contains: parameters (NDArrays), num_examples (int), metrics (dict).
        unpacked: List[Tuple[NDArrays, int, Dict[str, Scalar]]] = []
        client_metrics: List[Dict[str, Scalar]] = []

        for client_proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            unpacked.append((ndarrays, fit_res.num_examples, fit_res.metrics))
            client_metrics.append(fit_res.metrics)

        K = len(unpacked)
        logger.info(f"[Round {server_round}] Aggregating {K} client updates via FedCSS ...")

        # ── Step 2: Compute CSS scores and normalised weights ─────────────────
        #
        #   For each client i ∈ {0,...,K-1}:
        #       CSS_i = α·V̂_i + β·T̂_i + γ·R̂_i + δ·Ĉ_i
        #       w_i   = CSS_i / Σ_j CSS_j
        #
        css_scores, css_weights = self.css_calculator.compute(client_metrics)

        # Log the CSS diagnostics for interpretability
        logger.info(f"[Round {server_round}] CSS Diagnostics:")
        for i, (metrics, score, weight) in enumerate(
            zip(client_metrics, css_scores, css_weights)
        ):
            logger.info(
                f"  Client {int(metrics.get('client_id', i)):02d} | "
                f"vol={metrics.get('avg_traffic_volume', 0):.2f}  "
                f"var={metrics.get('temporal_variance', 0):.2f}  "
                f"rel={metrics.get('sensor_reliability_score', 0):.4f}  "
                f"cov={metrics.get('spatial_coverage_degree', 0):.4f}  "
                f"→ CSS={score:.4f}  w={weight:.4f}"
            )

        # ── Step 3: CSS-weighted parameter aggregation ────────────────────────
        #
        #   Δθ_l^* = Σ_i w_i · Δθ_l^(i)    for each adapter layer l
        #
        aggregated_ndarrays = css_weighted_aggregate(unpacked, css_weights)

        # ── Step 4: Save the global adapter to disk ───────────────────────────
        checkpoint_path = self._save_global_adapter(aggregated_ndarrays, server_round)

        # ── Step 5: Compute round-level aggregated metrics ────────────────────
        total_examples = sum(n for _, n, _ in unpacked)

        # Weighted average of train_loss using CSS weights (not raw sample counts)
        weighted_loss = float(
            sum(
                css_weights[i] * float(m.get("train_loss", 0.0))
                for i, (_, _, m) in enumerate(unpacked)
            )
        )

        elapsed = round(time.time() - t0, 3)

        aggregated_metrics: Dict[str, Scalar] = {
            "round": server_round,
            "num_clients": K,
            "total_examples": total_examples,
            "css_weighted_train_loss": round(weighted_loss, 6),
            "aggregation_time_s": elapsed,
            # CSS weight distribution statistics
            "css_weight_min": round(float(css_weights.min()), 6),
            "css_weight_max": round(float(css_weights.max()), 6),
            "css_weight_std": round(float(css_weights.std()), 6),
            # Effective number of clients (1 / Σ w_i²) — diversity measure
            # = K when all clients have equal weight, < K when some dominate.
            "css_effective_clients": round(float(1.0 / (css_weights ** 2).sum()), 4),
        }

        # Store history for offline analysis
        self.round_history.append({
            "round": server_round,
            "css_scores": css_scores.tolist(),
            "css_weights": css_weights.tolist(),
            "metrics": aggregated_metrics,
            "checkpoint": checkpoint_path,
        })

        # Persist round history JSON
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.round_history, f, indent=2)

        logger.info(
            f"[Round {server_round}] Aggregation complete | "
            f"CSS-loss={weighted_loss:.4f} | "
            f"eff_clients={aggregated_metrics['css_effective_clients']:.2f}/{K} | "
            f"elapsed={elapsed}s"
        )

        return ndarrays_to_parameters(aggregated_ndarrays), aggregated_metrics

    # ── Optional Override: aggregate_evaluate ────────────────────────────────

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregates distributed evaluation results using CSS weights.
        Falls back to FedAvg's sample-count weighting if client metrics are
        unavailable in EvaluateRes.
        """
        if not results:
            return None, {}

        # Weighted average evaluation loss — use sample-count weighting here
        # (clients don't return CSS metrics in evaluate(), only in fit())
        total_examples = sum(r.num_examples for _, r in results)
        if total_examples == 0:
            return None, {}

        weighted_eval_loss = sum(
            (r.num_examples / total_examples) * r.loss
            for _, r in results
        )

        eval_metrics: Dict[str, Scalar] = {
            "round": server_round,
            "eval_loss": round(float(weighted_eval_loss), 6),
            "num_eval_clients": len(results),
        }

        logger.info(
            f"[Round {server_round}] Evaluation | "
            f"loss={weighted_eval_loss:.4f} | clients={len(results)}"
        )
        return float(weighted_eval_loss), eval_metrics


# =============================================================================
# 4. FedCSS Simulation Driver (no real network needed)
# =============================================================================

class FedCSSSimulator:
    """
    Simulates multiple federated rounds of FedCSS aggregation using synthetic
    client results — no actual Flower gRPC server or network connection needed.

    Used for:
      • Unit-testing the FedCSS mathematics before deploying with real clients.
      • Profiling CSS weight distributions across different client configurations.
      • Generating round-history artifacts for analysis and publication.
    """

    def __init__(
        self,
        num_clients: int = 4,
        num_rounds: int = 5,
        num_adapter_tensors: int = 14,   # typical for 7 LoRA layers × 2 (A, B)
        adapter_dim: Tuple[int, int] = (16, 64),
        checkpoint_dir: str = "./checkpoints/global_adapter",
        css_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        rng_seed: int = 42,
    ):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.num_adapter_tensors = num_adapter_tensors
        self.adapter_dim = adapter_dim
        self.checkpoint_dir = checkpoint_dir
        self.rng = np.random.RandomState(rng_seed)

        self.strategy = FedCSS(
            css_alpha=css_weights[0],
            css_beta=css_weights[1],
            css_gamma=css_weights[2],
            css_delta=css_weights[3],
            checkpoint_dir=checkpoint_dir,
            fraction_fit=1.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
        ) if HAS_FLWR else None

        self.css_calculator = CSSWeightCalculator(*css_weights)

    def _make_synthetic_client_results(
        self,
        round_num: int,
    ) -> List[Tuple[NDArrays, int, Dict[str, Scalar]]]:
        """Generates synthetic client parameter updates and CSS metrics."""
        results = []
        for i in range(self.num_clients):
            # Simulate LoRA adapter deltas (small perturbations around zero)
            ndarrays = [
                self.rng.normal(0.0, 0.01, size=self.adapter_dim).astype(np.float32)
                for _ in range(self.num_adapter_tensors)
            ]
            num_examples = int(self.rng.randint(800, 1200))

            # Heterogeneous CSS signals per client (simulate non-IID traffic)
            metrics: Dict[str, Scalar] = {
                "client_id": i,
                "train_loss": float(self.rng.uniform(1.5, 2.5)),
                # Client 0 → high-volume urban freeway (SF Bay Area)
                # Client 1 → moderate suburban corridor
                # Client 2 → low-volume rural segment
                # Client 3 → high-coverage interchange hub
                "avg_traffic_volume": float([450.0, 250.0, 80.0, 320.0][i % 4]
                                             + self.rng.normal(0, 10)),
                "temporal_variance":  float([180.0, 95.0,  20.0, 140.0][i % 4]
                                             + self.rng.normal(0, 5)),
                "sensor_reliability_score": float([0.98, 0.92, 0.75, 0.88][i % 4]),
                "spatial_coverage_degree":  float([0.04, 0.03, 0.01, 0.06][i % 4]),
            }
            results.append((ndarrays, num_examples, metrics))
        return results

    def run(self) -> List[Dict[str, Any]]:
        """
        Executes the configured number of simulated federated rounds and
        returns the round history list.
        """
        logger.info(
            f"Starting FedCSS simulation: {self.num_rounds} rounds, "
            f"{self.num_clients} clients, {self.num_adapter_tensors} adapter tensors."
        )

        history = []
        # Track current global adapter (starts as zeros)
        global_adapter: NDArrays = [
            np.zeros(self.adapter_dim, dtype=np.float32)
            for _ in range(self.num_adapter_tensors)
        ]

        for r in range(1, self.num_rounds + 1):
            logger.info(f"\n{'='*60}\n[Round {r}/{self.num_rounds}]\n{'='*60}")

            client_results = self._make_synthetic_client_results(round_num=r)
            client_metrics = [m for _, _, m in client_results]
            css_scores, css_weights = self.css_calculator.compute(client_metrics)

            # Perform CSS-weighted aggregation
            aggregated = css_weighted_aggregate(client_results, css_weights)

            # Save checkpoint via strategy helper (or directly if no flwr)
            if self.strategy is not None:
                round_dir = self.strategy._save_global_adapter(aggregated, r)
            else:
                round_dir = os.path.join(self.checkpoint_dir, f"round_{r:04d}")
                os.makedirs(round_dir, exist_ok=True)
                for ti, arr in enumerate(aggregated):
                    np.save(os.path.join(round_dir, f"adapter_layer_{ti:04d}.npy"), arr)

            # Weighted training loss
            weighted_loss = sum(
                float(css_weights[i]) * float(m.get("train_loss", 0.0))
                for i, (_, _, m) in enumerate(client_results)
            )
            effective_clients = float(1.0 / (css_weights ** 2).sum())

            round_record = {
                "round": r,
                "css_scores": css_scores.tolist(),
                "css_weights": css_weights.tolist(),
                "css_weighted_loss": round(weighted_loss, 6),
                "effective_clients": round(effective_clients, 4),
                "checkpoint_dir": round_dir,
            }
            history.append(round_record)

            logger.info(f"[Round {r}] CSS Weights: {[round(w, 4) for w in css_weights.tolist()]}")
            logger.info(f"[Round {r}] Effective clients (diversity): {effective_clients:.2f}/{self.num_clients}")
            logger.info(f"[Round {r}] CSS-weighted loss: {weighted_loss:.4f}")
            logger.info(f"[Round {r}] Checkpoint: {round_dir}")

            global_adapter = aggregated

        # Save history JSON
        history_path = os.path.join(self.checkpoint_dir, "simulation_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info(f"\nSimulation history saved → {history_path}")

        return history


# =============================================================================
# 5. Real Flower Server Start Function
# =============================================================================

def start_fedcss_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    num_clients: int = 4,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.5,
    checkpoint_dir: str = "./checkpoints/global_adapter",
    css_alpha: float = 0.25,
    css_beta:  float = 0.25,
    css_gamma: float = 0.25,
    css_delta: float = 0.25,
) -> None:
    """
    Starts the Flower gRPC server with the FedCSS aggregation strategy.

    Clients connect via:
        fl.client.start_numpy_client(server_address="<host>:8080", client=...)

    Or the virtual-client engine can be used for local simulation:
        fl.simulation.start_simulation(client_fn=..., num_clients=4, ...)
    """
    if not HAS_FLWR:
        raise RuntimeError(
            "flwr is not installed. Run: pip install flwr"
        )

    strategy = FedCSS(
        css_alpha=css_alpha,
        css_beta=css_beta,
        css_gamma=css_gamma,
        css_delta=css_delta,
        checkpoint_dir=checkpoint_dir,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=num_clients,
        min_evaluate_clients=max(1, int(num_clients * fraction_evaluate)),
        min_available_clients=num_clients,
    )

    logger.info(f"Starting FedCSS Flower server at {server_address} ...")
    logger.info(f"  Rounds: {num_rounds} | Clients: {num_clients} | "
                f"CSS: α={css_alpha} β={css_beta} γ={css_gamma} δ={css_delta}")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


# =============================================================================
# 6. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedLLM Phase 5: FedCSS Federated Server"
    )
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./checkpoints/global_adapter")
    parser.add_argument("--css_alpha", type=float, default=0.25,
                        help="CSS weight for avg_traffic_volume")
    parser.add_argument("--css_beta",  type=float, default=0.25,
                        help="CSS weight for temporal_variance")
    parser.add_argument("--css_gamma", type=float, default=0.25,
                        help="CSS weight for sensor_reliability_score")
    parser.add_argument("--css_delta", type=float, default=0.25,
                        help="CSS weight for spatial_coverage_degree")
    parser.add_argument("--simulate", action="store_true",
                        help="Run a local FedCSS simulation (no network needed)")
    parser.add_argument("--sim_rounds", type=int, default=5,
                        help="Number of simulation rounds (--simulate only)")
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    print("=" * 80)
    print("[FedLLM] Phase 5: FedCSS Federated Server")
    print("=" * 80)
    print(f"  CSS weights: alpha={args.css_alpha}  beta={args.css_beta}  "
          f"gamma={args.css_gamma}  delta={args.css_delta}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 80)

    if args.simulate or not HAS_FLWR:
        if not HAS_FLWR and not args.simulate:
            print("[INFO] 'flwr' package is not installed. Running in standalone FedCSS simulation mode...")
            print("       To run a live gRPC server with distributed clients, install Flower: pip install flwr\n")

        # ── Local simulation: no network / GPU needed ─────────────────────────
        print(f"[+] Running FedCSS local simulation ({args.sim_rounds} rounds, "
              f"{args.num_clients} clients) ...\n")

        sim = FedCSSSimulator(
            num_clients=args.num_clients,
            num_rounds=args.sim_rounds,
            checkpoint_dir=args.checkpoint_dir,
            css_weights=(args.css_alpha, args.css_beta,
                         args.css_gamma, args.css_delta),
        )
        history = sim.run()

        print("\n[+] Simulation Summary:")
        print(f"  {'Round':<8} {'CSS-Loss':<14} {'EffClients':<14} {'CSS Weights'}")
        print(f"  {'-'*70}")
        for rec in history:
            weights_str = "  ".join(f"{w:.3f}" for w in rec["css_weights"])
            print(
                f"  {rec['round']:<8} "
                f"{rec['css_weighted_loss']:<14.4f} "
                f"{rec['effective_clients']:<14.2f} "
                f"[{weights_str}]"
            )

        print(f"\n[SUCCESS] Phase 5: FedCSS simulation complete!")
        print(f"  Checkpoints saved → {args.checkpoint_dir}")

    else:
        # ── Production mode: start gRPC Flower server ─────────────────────────
        start_fedcss_server(
            server_address=args.server_address,
            num_rounds=args.num_rounds,
            num_clients=args.num_clients,
            checkpoint_dir=args.checkpoint_dir,
            css_alpha=args.css_alpha,
            css_beta=args.css_beta,
            css_gamma=args.css_gamma,
            css_delta=args.css_delta,
        )

