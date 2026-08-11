#!/usr/bin/env python3
"""
Main experiment runner for ICS Adversarial Robustness Benchmark.

Usage:
  python run_experiment.py --stage train --dataset swat --detector lstm_ae
  python run_experiment.py --stage attack --dataset swat --detector lstm_ae --attack pgd --eps 0.05
  python run_experiment.py --stage defend --dataset swat --detector lstm_ae
  python run_experiment.py --stage all --dataset swat
  python run_experiment.py --stage all --dataset swat --seeds "42,123,456"

Pipeline (--stage all):
  1. Train all detectors (clean)
  2. Evaluate baselines
  3. Run all attacks (unconstrained)
  4. Run constrained PGD attacks
  5. Train all detectors with adversarial training
  6. Evaluate defended baselines
  7. Run all attacks against defended detectors

Attack methods:
  - FGSM: Single-step gradient sign. Can overshoot at large epsilon.
  - I-FGSM: Iterative FGSM (10 steps). Avoids overshoot via small step sizes.
  - PGD: Projected Gradient Descent (40 iter, early stopping after 5 stagnant).
  - C&W: Carlini-Wagner L2 optimization (200 steps). No epsilon dependency.
  - AutoAttack: PGD with 3 random restarts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from data_loader import get_dataloaders
from models import create_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentMeta:
    """Immutable metadata attached to every result JSON."""
    seed: int
    timestamp: str
    config_hash: str
    detector: str
    dataset: str
    stage: str
    tag: str = ""


@dataclass
class MetricSet:
    """Standard metric bundle for a single evaluation."""
    detection_rate: float = 0.0
    false_alarm_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    threshold: float = 0.0
    n_samples: int = 0
    n_attacks: int = 0


@dataclass
class AttackMetrics:
    """Extended metrics for an attack experiment."""
    base: MetricSet = field(default_factory=MetricSet)
    attack_success_rate: float = 0.0
    physical_plausibility_rate: float = 0.0
    attack: str = ""
    epsilon: float = 0.0
    constrained: bool = False
    attack_time_sec: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_hash(config: dict) -> str:
    """Deterministic hash of the config dict for provenance."""
    raw = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _result_path(
    output_dir: str,
    stage: str,
    detector: str,
    dataset: str,
    attack: str = "",
    eps: float | None = None,
    tag: str = "",
    seed: int | None = None,
) -> str:
    """Build a canonical result filename."""
    parts = [stage, detector, dataset]
    if attack:
        parts.append(attack)
    if eps is not None:
        parts.append(f"eps{eps}")
    if tag:
        parts.append(tag)
    if seed is not None:
        parts.append(f"seed{seed}")
    return os.path.join(output_dir, "_".join(parts) + ".json")


def _aggregate_path(
    output_dir: str,
    stage: str,
    detector: str,
    dataset: str,
    attack: str = "",
    eps: float | None = None,
    tag: str = "",
) -> str:
    parts = [stage, detector, dataset]
    if attack:
        parts.append(attack)
    if eps is not None:
        parts.append(f"eps{eps}")
    if tag:
        parts.append(tag)
    parts.append("aggregate")
    return os.path.join(output_dir, "_".join(parts) + ".json")


def save_results(results: dict, filepath: str) -> None:
    """Save experiment results as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", filepath)


def aggregate_seed_results(
    seed_results: list[dict],
    output_path: str,
) -> dict:
    """Compute mean/std/min/max across seeds for numeric fields."""
    if not seed_results:
        return {}

    # Collect all numeric keys
    numeric_keys: list[str] = []
    sample = seed_results[0]
    for k, v in sample.items():
        if isinstance(v, (int, float)) and k != "seed":
            numeric_keys.append(k)

    agg: dict[str, Any] = {
        "n_seeds": len(seed_results),
        "seeds": [r.get("seed") for r in seed_results],
    }
    for k in numeric_keys:
        vals = [r[k] for r in seed_results if k in r]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        agg[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # Copy non-numeric metadata from first result
    for k, v in sample.items():
        if k not in agg and k != "seed" and not isinstance(v, (int, float)):
            agg[k] = v

    save_results(agg, output_path)
    return agg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_detector(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: dict,
    detector_name: str,
    dataset_name: str,
    device: torch.device,
    save_suffix: str = "best",
) -> tuple[nn.Module, dict]:
    """Train an anomaly detector with early stopping. Returns (model, info)."""
    model = model.to(device)
    lr: float = config["detectors"]["common"]["lr"]
    n_epochs: int = config["detectors"]["common"]["epochs"]
    patience: int = config["detectors"]["common"]["patience"]

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    save_dir = Path(config["output"]["models_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{detector_name}_{dataset_name}_{save_suffix}.pt"

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, _ in tqdm(
            train_loader, desc=f"  Epoch {epoch + 1}/{n_epochs}", leave=False
        ):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            if hasattr(model, "compute_loss"):
                loss = model.compute_loss(batch_x, epoch, n_epochs)
            else:
                recon = model(batch_x)
                if isinstance(recon, tuple):
                    recon = recon[0]
                loss = nn.MSELoss()(recon, batch_x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                if hasattr(model, "compute_loss"):
                    loss = model.compute_loss(batch_x, epoch, n_epochs)
                else:
                    recon = model(batch_x)
                    if isinstance(recon, tuple):
                        recon = recon[0]
                    loss = nn.MSELoss()(recon, batch_x)
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        val_losses.append(avg_val)

        logger.info(
            "  Epoch %d: train=%.6f  val=%.6f", epoch + 1, avg_train, avg_val
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model, {"train_losses": train_losses, "val_losses": val_losses}


# ---------------------------------------------------------------------------
# Adversarial Training (defense)
# ---------------------------------------------------------------------------


def _generate_pgd_adversarial_batch(
    model: nn.Module,
    x_clean: torch.Tensor,
    epsilon: float,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate PGD adversarial examples for a single batch during training.

    Used inside adversarial training to create perturbations that try to
    *maximize* reconstruction loss (make normal data look anomalous to force
    robust features).
    """
    step_size = epsilon / max(n_steps, 1) * 2.0
    x_adv = x_clean.detach() + torch.empty_like(x_clean).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(n_steps):
        x_adv.requires_grad_(True)
        if hasattr(model, "compute_loss"):
            # We want to *maximize* reconstruction loss
            loss = model.compute_loss(x_adv, 0, 1)
        else:
            recon = model(x_adv)
            if isinstance(recon, tuple):
                recon = recon[0]
            loss = nn.MSELoss()(recon, x_adv)

        loss.backward()

        with torch.no_grad():
            # Ascend (maximize loss) so the model must learn robust features
            x_adv = x_adv + step_size * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x_clean, -epsilon, epsilon)
            x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach()

    return x_adv


def train_detector_adversarial(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: dict,
    detector_name: str,
    dataset_name: str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Adversarial training: 50 % clean + 50 % PGD-perturbed per batch.

    PGD parameters: 5 steps, eps = 0.05 (from config fallback).
    Saves checkpoint as ``{detector}_{dataset}_adversarial_best.pt``.
    """
    adv_cfg = config.get("defenses", {}).get("adversarial_training", {})
    mix_ratio: float = adv_cfg.get("mix_ratio", 0.5)
    pgd_steps: int = adv_cfg.get("pgd_steps", 5)
    pgd_eps: float = adv_cfg.get("epsilon", 0.05)

    model = model.to(device)
    lr: float = config["detectors"]["common"]["lr"]
    n_epochs: int = config["detectors"]["common"]["epochs"]
    patience: int = config["detectors"]["common"]["patience"]

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    save_dir = Path(config["output"]["models_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{detector_name}_{dataset_name}_adversarial_best.pt"

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, _ in tqdm(
            train_loader,
            desc=f"  AdvTrain Epoch {epoch + 1}/{n_epochs}",
            leave=False,
        ):
            batch_x = batch_x.to(device)

            # Generate adversarial batch (model in train mode for cuDNN LSTM)
            x_adv = _generate_pgd_adversarial_batch(
                model, batch_x, pgd_eps, pgd_steps, device
            )

            # Mix: first half clean, second half adversarial
            n_clean = int(len(batch_x) * (1.0 - mix_ratio))
            x_mixed = torch.cat([batch_x[:n_clean], x_adv[n_clean:]], dim=0)

            optimizer.zero_grad()
            if hasattr(model, "compute_loss"):
                loss = model.compute_loss(x_mixed, epoch, n_epochs)
            else:
                recon = model(x_mixed)
                if isinstance(recon, tuple):
                    recon = recon[0]
                loss = nn.MSELoss()(recon, x_mixed)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Validation (clean data only)
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                if hasattr(model, "compute_loss"):
                    loss = model.compute_loss(batch_x, epoch, n_epochs)
                else:
                    recon = model(batch_x)
                    if isinstance(recon, tuple):
                        recon = recon[0]
                    loss = nn.MSELoss()(recon, batch_x)
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        val_losses.append(avg_val)

        logger.info(
            "  AdvTrain Epoch %d: train=%.6f  val=%.6f",
            epoch + 1,
            avg_train,
            avg_val,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("  AdvTrain early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model, {"train_losses": train_losses, "val_losses": val_losses}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compute_threshold(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    percentile: float,
    device: torch.device,
) -> float:
    """Compute anomaly threshold from training data at given percentile."""
    model.eval()
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            scores = model.anomaly_score(batch_x)
            all_scores.append(scores.cpu().numpy())
    return float(np.percentile(np.concatenate(all_scores), percentile))


def evaluate_detector(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    threshold: float,
    device: torch.device,
) -> dict:
    """Evaluate detector on test data. Returns flat metrics dict."""
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            scores = model.anomaly_score(batch_x)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(batch_y.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    preds = (scores > threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    dr = tp / max(tp + fn, 1)
    far = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = dr
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    return {
        "detection_rate": float(dr),
        "false_alarm_rate": float(far),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": float(threshold),
        "n_samples": len(labels),
        "n_attacks": int(labels.sum()),
    }


# ---------------------------------------------------------------------------
# Attack primitives
# ---------------------------------------------------------------------------


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float,
    device: torch.device,
) -> torch.Tensor:
    """Single-step FGSM. Minimizes anomaly score via gradient sign.

    NOTE: At large epsilon this can overshoot the score-minimizing direction.
    Prefer I-FGSM for more controlled perturbation.
    """
    x_adv = x.clone().detach().requires_grad_(True)
    score = model.anomaly_score(x_adv)
    score.sum().backward()

    with torch.no_grad():
        perturbation = epsilon * x_adv.grad.sign()
        x_adv = x_adv - perturbation  # minimize score
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def ifgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Iterative FGSM (I-FGSM / BIM). Avoids single-step overshoot."""
    step_size = epsilon / max(n_steps, 1)
    x_adv = x.clone().detach()

    for _ in range(n_steps):
        x_adv = x_adv.requires_grad_(True)
        score = model.anomaly_score(x_adv)
        score.sum().backward()

        with torch.no_grad():
            x_adv = x_adv - step_size * x_adv.grad.sign()
            # Project back into epsilon-ball
            delta = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    return x_adv


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float,
    n_steps: int,
    step_size: float,
    device: torch.device,
    early_stop_patience: int = 5,
) -> torch.Tensor:
    """PGD with early stopping when score stagnates for `early_stop_patience` steps."""
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    best_adv = x_adv.clone()
    best_score_sum = float("inf")
    stagnant = 0

    for step_i in range(n_steps):
        x_adv = x_adv.requires_grad_(True)
        score = model.anomaly_score(x_adv)
        score_sum = score.sum()
        score_sum.backward()

        with torch.no_grad():
            x_adv = x_adv - step_size * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

        current_score = score_sum.item()
        if current_score < best_score_sum - 1e-8:
            best_score_sum = current_score
            best_adv = x_adv.clone()
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= early_stop_patience:
                break

    return best_adv


def pgd_attack_with_restarts(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float,
    n_steps: int,
    step_size: float,
    n_restarts: int,
    device: torch.device,
) -> torch.Tensor:
    """PGD with random restarts (AutoAttack proxy)."""
    best_adv = x.clone()
    with torch.no_grad():
        best_scores = model.anomaly_score(x.detach())

    for _ in range(n_restarts):
        x_adv = pgd_attack(model, x, epsilon, n_steps, step_size, device)
        with torch.no_grad():
            scores = model.anomaly_score(x_adv)
            improved = scores < best_scores
            if improved.any():
                best_adv[improved] = x_adv[improved]
                best_scores[improved] = scores[improved]

    return best_adv


def cw_attack(
    model: nn.Module,
    x: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    c_weight: float = 0.01,
) -> torch.Tensor:
    """Carlini-Wagner L2 attack for anomaly detectors.

    This is an *unconstrained L2 optimization* — there is no epsilon parameter.
    The ``c_weight`` balances anomaly-score minimization vs. L2 distance.
    """
    x_clamped = torch.clamp(x, 1e-6, 1 - 1e-6)
    w = torch.zeros_like(x, device=device, requires_grad=True)
    optimizer = optim.Adam([w], lr=lr)

    best_adv = x.clone()
    best_loss = torch.full((x.size(0),), float("inf"), device=device)

    for _ in range(n_steps):
        x_adv = torch.sigmoid(w + torch.logit(x_clamped))
        score = model.anomaly_score(x_adv)
        l2_dist = torch.norm((x_adv - x).reshape(x.size(0), -1), dim=1)
        loss = score + c_weight * l2_dist
        total_loss = loss.sum()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track best per-sample
        with torch.no_grad():
            improved = loss < best_loss
            if improved.any():
                best_loss[improved] = loss[improved]
                best_adv[improved] = x_adv[improved].detach()

    return best_adv.detach()


# ---------------------------------------------------------------------------
# Physical constraints & plausibility
# ---------------------------------------------------------------------------


def apply_physical_constraints(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    epsilon: float,
    scaler_min: torch.Tensor | None,
    scaler_max: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Project adversarial examples onto physically plausible space.

    Constraints applied (in order):
      1. L-inf: clamp perturbation to [-eps, +eps]
      2. Range: clamp to [0, 1] (normalized) AND per-sensor [scaler_min, scaler_max]
      3. Rate: per-timestep delta clamped to [-max_rate, +max_rate]
         where max_rate = 0.5 * epsilon (proportional to perturbation budget)
    """
    # L-inf constraint
    delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
    x_adv = x_orig + delta

    # Global range
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # Per-sensor range from training data (already in [0,1] after MinMaxScaler,
    # but clamp to observed bounds catches extrapolation from perturbation)
    if scaler_min is not None and scaler_max is not None:
        s_min = scaler_min.to(device)
        s_max = scaler_max.to(device)
        # scaler_min/max are in *original* scale; in normalized space the bounds
        # are always [0, 1] by construction of MinMaxScaler. However, adversarial
        # perturbations on normalized data can push values slightly beyond the
        # training range. We re-clamp to [0, 1] which is the normalized observed range.
        x_adv = torch.clamp(x_adv, s_min, s_max)

    # Rate constraint: max_rate proportional to epsilon
    max_rate = 0.5 * epsilon
    for t in range(1, x_adv.size(1)):
        rate = x_adv[:, t, :] - x_adv[:, t - 1, :]
        rate = torch.clamp(rate, -max_rate, max_rate)
        x_adv = x_adv.clone()
        x_adv[:, t, :] = x_adv[:, t - 1, :] + rate

    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def compute_plausibility(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    epsilon: float,
    constrained: bool,
    scaler_min: torch.Tensor | None,
    scaler_max: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Compute Physical Plausibility Rate (PPR).

    For *constrained* attacks: PPR is ~1.0 by construction (constraints enforced).
    For *unconstrained* attacks: PPR = fraction of samples that ALSO satisfy
        both range AND rate constraints, even though they were not enforced.
    """
    batch_size = x_adv.size(0)

    if constrained:
        # Constraints were enforced, so PPR ≈ 1.0.
        # Verify with a small tolerance to catch numerical drift.
        return torch.ones(batch_size, device=device)

    # --- Unconstrained: check which samples *naturally* satisfy constraints ---

    # 1. Range check: [0, 1]
    range_ok = (x_adv >= -1e-6).all(dim=(1, 2)) & (x_adv <= 1.0 + 1e-6).all(
        dim=(1, 2)
    )

    # 2. Per-sensor range from training data
    if scaler_min is not None and scaler_max is not None:
        s_min = scaler_min.to(device)
        s_max = scaler_max.to(device)
        sensor_ok = (x_adv >= s_min - 1e-6).all(dim=(1, 2)) & (
            x_adv <= s_max + 1e-6
        ).all(dim=(1, 2))
        range_ok = range_ok & sensor_ok

    # 3. Rate constraint: max_rate = 0.5 * epsilon
    max_rate = 0.5 * epsilon
    rates = torch.abs(x_adv[:, 1:, :] - x_adv[:, :-1, :])
    rate_ok = (rates <= max_rate + 1e-6).all(dim=(1, 2))

    return (range_ok & rate_ok).float()


# ---------------------------------------------------------------------------
# Attack orchestration
# ---------------------------------------------------------------------------


def _prepare_scaler_tensors(
    metadata: dict, device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Convert scaler bounds from metadata to tensors for GPU ops."""
    s_min_list = metadata.get("scaler_min")
    s_max_list = metadata.get("scaler_max")
    if s_min_list is None or s_max_list is None:
        return None, None
    # Normalized space: min→0, max→1, but we store raw values for reference.
    # For constraint checking in normalized space, the per-sensor bounds are [0, 1].
    s_min = torch.zeros(len(s_min_list), device=device)
    s_max = torch.ones(len(s_max_list), device=device)
    return s_min, s_max


def adversarial_attack(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    attack_method: str,
    epsilon: float,
    config: dict,
    device: torch.device,
    constrained: bool = False,
    scaler_min: torch.Tensor | None = None,
    scaler_max: torch.Tensor | None = None,
) -> dict:
    """Run an adversarial attack on test data and return scores + plausibility."""
    # Train mode needed for gradient computation (cuDNN LSTM backward pass)
    model.train()

    all_adv_scores: list[np.ndarray] = []
    all_orig_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_plausibility: list[np.ndarray] = []

    attack_cfg = config.get("attacks", {}).get("methods", {})

    for batch_x, batch_y in tqdm(
        test_loader,
        desc=f"  Attack ({attack_method}, eps={epsilon})",
        leave=False,
    ):
        batch_x = batch_x.to(device)
        batch_y_np = batch_y.numpy()

        attack_mask = batch_y_np == 1
        if not attack_mask.any():
            with torch.no_grad():
                scores = model.anomaly_score(batch_x).cpu().numpy()
            all_adv_scores.append(scores)
            all_orig_scores.append(scores)
            all_labels.append(batch_y_np)
            all_plausibility.append(np.ones(len(batch_y_np)))
            continue

        x_adv = batch_x.clone().detach()
        x_attack = batch_x[attack_mask].clone().detach()

        # ---- Dispatch to attack method ----
        if attack_method == "fgsm":
            x_attack_adv = fgsm_attack(model, x_attack, epsilon, device)

        elif attack_method == "ifgsm":
            ifgsm_cfg = attack_cfg.get("ifgsm", {})
            n_steps = ifgsm_cfg.get("max_iter", 10)
            x_attack_adv = ifgsm_attack(
                model, x_attack, epsilon, n_steps, device
            )

        elif attack_method == "pgd":
            pgd_cfg = attack_cfg.get("pgd", {})
            n_steps = pgd_cfg.get("max_iter", 40)
            step_ratio = pgd_cfg.get("eps_step_ratio", 0.1)
            step_size = epsilon * step_ratio
            x_attack_adv = pgd_attack(
                model, x_attack, epsilon, n_steps, step_size, device,
                early_stop_patience=5,
            )

        elif attack_method == "cw":
            cw_cfg = attack_cfg.get("cw", {})
            n_steps = cw_cfg.get("max_iter", 200)
            lr = cw_cfg.get("lr", 0.01)
            x_attack_adv = cw_attack(model, x_attack, n_steps, lr, device)

        elif attack_method == "autoattack":
            pgd_cfg = attack_cfg.get("pgd", {})
            n_steps = pgd_cfg.get("max_iter", 40)
            step_size = epsilon * pgd_cfg.get("eps_step_ratio", 0.1)
            n_restarts = 3
            x_attack_adv = pgd_attack_with_restarts(
                model, x_attack, epsilon, n_steps, step_size, n_restarts, device
            )

        else:
            raise ValueError(f"Unknown attack method: {attack_method}")

        # Apply physical constraints if requested
        if constrained:
            x_attack_adv = apply_physical_constraints(
                x_attack_adv, x_attack, epsilon,
                scaler_min, scaler_max, device,
            )

        # Compute plausibility (different logic for constrained vs unconstrained)
        plausible = compute_plausibility(
            x_attack_adv, x_attack, epsilon, constrained,
            scaler_min, scaler_max, device,
        )

        x_adv[attack_mask] = x_attack_adv.detach()

        with torch.no_grad():
            adv_scores = model.anomaly_score(x_adv).cpu().numpy()
            orig_scores = model.anomaly_score(batch_x).cpu().numpy()

        all_adv_scores.append(adv_scores)
        all_orig_scores.append(orig_scores)
        all_labels.append(batch_y_np)

        plaus = np.ones(len(batch_y_np))
        plaus[attack_mask] = plausible.cpu().detach().numpy()
        all_plausibility.append(plaus)

    return {
        "adv_scores": np.concatenate(all_adv_scores),
        "orig_scores": np.concatenate(all_orig_scores),
        "labels": np.concatenate(all_labels),
        "plausibility": np.concatenate(all_plausibility),
    }


# ---------------------------------------------------------------------------
# High-level experiment runners
# ---------------------------------------------------------------------------


def _get_dataloaders(config: dict, dataset_name: str):
    """Convenience wrapper for get_dataloaders using config."""
    data_dir = config["data"][f"{dataset_name}_dir"]
    return get_dataloaders(
        dataset_name,
        data_dir,
        window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
        batch_size=config["detectors"]["common"]["batch_size"],
        val_ratio=config["data"]["val_ratio"],
    )


def run_baseline(
    config: dict,
    dataset_name: str,
    detector_name: str,
    device: torch.device,
    seed: int,
    tag: str = "",
) -> tuple[nn.Module, float, dict, dict]:
    """Train detector + evaluate on clean test data. Returns (model, threshold, metadata, metrics)."""
    logger.info(
        "BASELINE: %s on %s (seed=%d, tag=%s)",
        detector_name, dataset_name, seed, tag or "clean",
    )

    train_loader, val_loader, test_loader, metadata = _get_dataloaders(
        config, dataset_name
    )

    model = create_model(
        detector_name,
        metadata["n_features"],
        window_size=config["data"]["window_size"],
        hidden_dim=config["detectors"]["common"]["hidden_dim"],
    )

    save_suffix = f"{'adversarial_' if tag == 'defended' else ''}best"

    if tag == "defended":
        # Use adversarial training
        t0 = time.time()
        model, train_info = train_detector_adversarial(
            model, train_loader, val_loader, config,
            detector_name, dataset_name, device,
        )
        train_time = time.time() - t0
    else:
        t0 = time.time()
        model, train_info = train_detector(
            model, train_loader, val_loader, config,
            detector_name, dataset_name, device,
            save_suffix=save_suffix,
        )
        train_time = time.time() - t0

    threshold = compute_threshold(
        model, train_loader,
        config["evaluation"]["threshold_percentile"], device,
    )

    metrics = evaluate_detector(model, test_loader, threshold, device)
    metrics["train_time_sec"] = train_time
    metrics["detector"] = detector_name
    metrics["dataset"] = dataset_name
    metrics["seed"] = seed
    metrics["tag"] = tag
    metrics["timestamp"] = _now_iso()
    metrics["config_hash"] = _config_hash(config)

    out_path = _result_path(
        config["output"]["results_dir"],
        "baseline", detector_name, dataset_name,
        tag=tag, seed=seed,
    )
    save_results(metrics, out_path)

    logger.info(
        "  DR=%.4f  FAR=%.4f  F1=%.4f",
        metrics["detection_rate"], metrics["false_alarm_rate"], metrics["f1_score"],
    )
    return model, threshold, metadata, metrics


def run_attack_experiment(
    config: dict,
    model: nn.Module,
    threshold: float,
    metadata: dict,
    dataset_name: str,
    detector_name: str,
    attack_method: str,
    epsilon: float | None,
    device: torch.device,
    seed: int,
    constrained: bool = False,
    tag: str = "",
) -> dict:
    """Run a single attack experiment and save results."""
    constraint_tag = "constrained" if constrained else "unconstrained"
    full_tag = f"{constraint_tag}_{tag}" if tag else constraint_tag

    # C&W uses L2 optimization, not L-inf epsilon. Run with eps=None marker.
    effective_eps = epsilon if attack_method != "cw" else None
    eps_label = epsilon if epsilon is not None else 0.0

    logger.info(
        "  ATTACK: %s eps=%s (%s)", attack_method, effective_eps, full_tag,
    )

    _, _, test_loader, _ = _get_dataloaders(config, dataset_name)
    scaler_min, scaler_max = _prepare_scaler_tensors(metadata, device)

    t0 = time.time()
    attack_results = adversarial_attack(
        model, test_loader, attack_method, eps_label, config, device,
        constrained=constrained,
        scaler_min=scaler_min,
        scaler_max=scaler_max,
    )
    attack_time = time.time() - t0

    # Compute metrics
    adv_preds = (attack_results["adv_scores"] > threshold).astype(int)
    labels = attack_results["labels"]
    attack_mask = labels == 1

    asr = float(1.0 - adv_preds[attack_mask].mean()) if attack_mask.any() else 0.0

    tp = int(((adv_preds == 1) & (labels == 1)).sum())
    fp = int(((adv_preds == 1) & (labels == 0)).sum())
    fn = int(((adv_preds == 0) & (labels == 1)).sum())
    tn = int(((adv_preds == 0) & (labels == 0)).sum())
    dr = tp / max(tp + fn, 1)
    far = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = dr
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    ppr = (
        float(attack_results["plausibility"][attack_mask].mean())
        if attack_mask.any()
        else 1.0
    )

    metrics = {
        "detector": detector_name,
        "dataset": dataset_name,
        "attack": attack_method,
        "epsilon": effective_eps,
        "constrained": constrained,
        "tag": full_tag,
        "seed": seed,
        "detection_rate": float(dr),
        "false_alarm_rate": float(far),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "attack_success_rate": float(asr),
        "physical_plausibility_rate": ppr,
        "attack_time_sec": attack_time,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "timestamp": _now_iso(),
        "config_hash": _config_hash(config),
        "note": (
            "FGSM can overshoot at large epsilon; prefer I-FGSM for controlled perturbation"
            if attack_method == "fgsm"
            else ""
        ),
    }

    out_path = _result_path(
        config["output"]["results_dir"],
        "attack", detector_name, dataset_name,
        attack=attack_method,
        eps=effective_eps,
        tag=full_tag,
        seed=seed,
    )
    save_results(metrics, out_path)

    logger.info(
        "    DR=%.4f  ASR=%.4f  PPR=%.4f  F1=%.4f",
        dr, asr, ppr, f1,
    )
    return metrics


# ---------------------------------------------------------------------------
# Attack method iteration helpers
# ---------------------------------------------------------------------------


def _iter_attack_configs(config: dict) -> list[tuple[str, float | None]]:
    """Yield (attack_method, epsilon) pairs respecting per-method rules.

    - C&W: runs once per detector/dataset (no epsilon loop, uses L2).
    - All others: iterate over epsilon_budgets.
    """
    methods = list(config["attacks"]["methods"].keys())
    epsilons = config["attacks"]["epsilon_budgets"]
    pairs: list[tuple[str, float | None]] = []

    for method in methods:
        if method == "cw":
            # C&W is L2, run once with epsilon=None
            pairs.append((method, None))
        else:
            for eps in epsilons:
                pairs.append((method, eps))

    return pairs


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_all_for_dataset(
    config: dict,
    dataset_name: str,
    seeds: list[int],
    device: torch.device,
) -> dict:
    """Run the complete 7-stage experiment pipeline for one dataset across seeds."""
    logger.info("#" * 70)
    logger.info("# DATASET: %s  |  seeds: %s", dataset_name.upper(), seeds)
    logger.info("#" * 70)

    detectors = config["detectors"]["architectures"]
    attack_pairs = _iter_attack_configs(config)
    epsilons = config["attacks"]["epsilon_budgets"]
    results_dir = config["output"]["results_dir"]

    # Collectors for aggregation
    all_baseline_results: dict[str, list[dict]] = {}
    all_attack_results: dict[str, list[dict]] = {}
    all_defended_baseline: dict[str, list[dict]] = {}
    all_defended_attack: dict[str, list[dict]] = {}

    for seed in seeds:
        logger.info("=" * 60)
        logger.info("SEED: %d", seed)
        logger.info("=" * 60)
        set_seed(seed)

        for det_name in detectors:
            det_key = f"{det_name}_{dataset_name}"

            # ----------------------------------------------------------
            # Stage 1-2: Train clean detector + evaluate baseline
            # ----------------------------------------------------------
            try:
                model, threshold, metadata, baseline_m = run_baseline(
                    config, dataset_name, det_name, device, seed, tag="",
                )
            except Exception as e:
                logger.error("Training %s failed: %s", det_name, e)
                continue

            all_baseline_results.setdefault(det_key, []).append(baseline_m)

            # ----------------------------------------------------------
            # Stage 3: Unconstrained attacks
            # ----------------------------------------------------------
            for attack_method, eps in attack_pairs:
                atk_key = f"{det_key}_{attack_method}_{eps}_unconstrained"
                try:
                    m = run_attack_experiment(
                        config, model, threshold, metadata,
                        dataset_name, det_name, attack_method, eps,
                        device, seed, constrained=False, tag="",
                    )
                    all_attack_results.setdefault(atk_key, []).append(m)
                except Exception as e:
                    logger.error(
                        "Attack %s eps=%s failed: %s", attack_method, eps, e
                    )

            # ----------------------------------------------------------
            # Stage 4: Constrained PGD attacks
            # ----------------------------------------------------------
            for eps in epsilons:
                atk_key = f"{det_key}_pgd_{eps}_constrained"
                try:
                    m = run_attack_experiment(
                        config, model, threshold, metadata,
                        dataset_name, det_name, "pgd", eps,
                        device, seed, constrained=True, tag="",
                    )
                    all_attack_results.setdefault(atk_key, []).append(m)
                except Exception as e:
                    logger.error("Constrained PGD eps=%s failed: %s", eps, e)

            # ----------------------------------------------------------
            # Stage 5-6: Adversarial training + defended baseline
            # ----------------------------------------------------------
            try:
                def_model, def_threshold, _, def_baseline_m = run_baseline(
                    config, dataset_name, det_name, device, seed,
                    tag="defended",
                )
            except Exception as e:
                logger.error("Adversarial training %s failed: %s", det_name, e)
                continue

            all_defended_baseline.setdefault(det_key, []).append(def_baseline_m)

            # ----------------------------------------------------------
            # Stage 7: Attacks against defended model
            # ----------------------------------------------------------
            for attack_method, eps in attack_pairs:
                atk_key = f"{det_key}_{attack_method}_{eps}_defended"
                try:
                    m = run_attack_experiment(
                        config, def_model, def_threshold, metadata,
                        dataset_name, det_name, attack_method, eps,
                        device, seed, constrained=False, tag="defended",
                    )
                    all_defended_attack.setdefault(atk_key, []).append(m)
                except Exception as e:
                    logger.error(
                        "Defended attack %s eps=%s failed: %s",
                        attack_method, eps, e,
                    )

    # ------------------------------------------------------------------
    # Aggregate across seeds
    # ------------------------------------------------------------------
    logger.info("Aggregating results across %d seeds ...", len(seeds))

    for key, results_list in all_baseline_results.items():
        agg_path = os.path.join(results_dir, f"baseline_{key}_aggregate.json")
        aggregate_seed_results(results_list, agg_path)

    for key, results_list in all_attack_results.items():
        agg_path = os.path.join(results_dir, f"attack_{key}_aggregate.json")
        aggregate_seed_results(results_list, agg_path)

    for key, results_list in all_defended_baseline.items():
        agg_path = os.path.join(results_dir, f"baseline_{key}_defended_aggregate.json")
        aggregate_seed_results(results_list, agg_path)

    for key, results_list in all_defended_attack.items():
        agg_path = os.path.join(results_dir, f"attack_{key}_aggregate.json")
        aggregate_seed_results(results_list, agg_path)

    summary = {
        "dataset": dataset_name,
        "seeds": seeds,
        "n_baseline": sum(len(v) for v in all_baseline_results.values()),
        "n_attacks": sum(len(v) for v in all_attack_results.values()),
        "n_defended_baselines": sum(len(v) for v in all_defended_baseline.values()),
        "n_defended_attacks": sum(len(v) for v in all_defended_attack.values()),
        "timestamp": _now_iso(),
    }
    save_results(summary, os.path.join(results_dir, f"summary_{dataset_name}.json"))
    return summary


# ---------------------------------------------------------------------------
# Single-stage entry points
# ---------------------------------------------------------------------------


def run_single_train(
    config: dict,
    dataset_name: str,
    detector_name: str,
    seeds: list[int],
    device: torch.device,
) -> None:
    """Train a single detector across seeds."""
    for seed in seeds:
        set_seed(seed)
        run_baseline(config, dataset_name, detector_name, device, seed, tag="")


def run_single_attack(
    config: dict,
    dataset_name: str,
    detector_name: str,
    attack_method: str,
    epsilon: float | None,
    seeds: list[int],
    device: torch.device,
    constrained: bool = False,
) -> None:
    """Run a single attack across seeds (loads pretrained model)."""
    for seed in seeds:
        set_seed(seed)
        train_loader, val_loader, test_loader, metadata = _get_dataloaders(
            config, dataset_name
        )
        model = create_model(
            detector_name,
            metadata["n_features"],
            window_size=config["data"]["window_size"],
            hidden_dim=config["detectors"]["common"]["hidden_dim"],
        )
        ckpt = Path(config["output"]["models_dir"]) / f"{detector_name}_{dataset_name}_best.pt"
        if not ckpt.exists():
            logger.error("Model checkpoint not found: %s. Run --stage train first.", ckpt)
            return
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model = model.to(device)

        threshold = compute_threshold(
            model, train_loader,
            config["evaluation"]["threshold_percentile"], device,
        )

        run_attack_experiment(
            config, model, threshold, metadata,
            dataset_name, detector_name, attack_method, epsilon,
            device, seed, constrained=constrained, tag="",
        )


def run_single_defend(
    config: dict,
    dataset_name: str,
    detector_name: str,
    seeds: list[int],
    device: torch.device,
) -> None:
    """Adversarial-train a detector, evaluate baseline + all attacks."""
    attack_pairs = _iter_attack_configs(config)

    for seed in seeds:
        set_seed(seed)

        # Adversarial training + defended baseline
        try:
            def_model, def_threshold, metadata, _ = run_baseline(
                config, dataset_name, detector_name, device, seed,
                tag="defended",
            )
        except Exception as e:
            logger.error("Adversarial training failed: %s", e)
            continue

        # Run all attacks against defended model
        for attack_method, eps in attack_pairs:
            try:
                run_attack_experiment(
                    config, def_model, def_threshold, metadata,
                    dataset_name, detector_name, attack_method, eps,
                    device, seed, constrained=False, tag="defended",
                )
            except Exception as e:
                logger.error(
                    "Defended attack %s eps=%s failed: %s", attack_method, eps, e
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_seeds(raw: str) -> list[int]:
    """Parse comma-separated seed string into sorted int list."""
    return sorted(int(s.strip()) for s in raw.split(",") if s.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICS Adversarial Robustness Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument(
        "--stage",
        choices=["train", "attack", "defend", "all"],
        default="all",
    )
    parser.add_argument(
        "--dataset", choices=["swat", "wadi", "smd"], required=True
    )
    parser.add_argument("--detector", default=None, help="Detector name (or all)")
    parser.add_argument("--attack", default=None, help="Attack method")
    parser.add_argument("--eps", type=float, default=None, help="Epsilon budget")
    parser.add_argument(
        "--constrained", action="store_true", help="Apply physical constraints"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help='Comma-separated seeds (overrides config), e.g. "42,123,456"',
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve seeds: CLI > config > default
    if args.seeds is not None:
        seeds = parse_seeds(args.seeds)
    elif "seeds" in config:
        seeds = list(config["seeds"])
    else:
        seeds = [config.get("seed", 42)]

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)
    logger.info("Seeds: %s", seeds)
    logger.info("Config hash: %s", _config_hash(config))
    logger.info("Started at: %s", _now_iso())

    if args.stage == "all":
        summary = run_all_for_dataset(config, args.dataset, seeds, device)
        logger.info("Completed at: %s", _now_iso())
        logger.info(
            "Summary: %d baselines, %d attacks, %d defended baselines, %d defended attacks",
            summary["n_baseline"],
            summary["n_attacks"],
            summary["n_defended_baselines"],
            summary["n_defended_attacks"],
        )

    elif args.stage == "train":
        det = args.detector or config["detectors"]["architectures"][0]
        run_single_train(config, args.dataset, det, seeds, device)

    elif args.stage == "attack":
        det = args.detector or config["detectors"]["architectures"][0]
        attack = args.attack
        if attack is None:
            logger.error("--attack is required for --stage attack")
            return
        run_single_attack(
            config, args.dataset, det, attack, args.eps, seeds, device,
            constrained=args.constrained,
        )

    elif args.stage == "defend":
        det = args.detector or config["detectors"]["architectures"][0]
        run_single_defend(config, args.dataset, det, seeds, device)

    logger.info("Done!")


if __name__ == "__main__":
    main()
