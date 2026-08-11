#!/usr/bin/env python3
"""
Experiment A: Within-Dataset Gradient Norm Analysis

Goal: Test if gradient norm is an ARCHITECTURAL effect (not dataset confound).
Method: Compute gradient norms for LSTM-AE, USAD, TranAD on SWaT using
        the SAME dataset, then check if norm ordering matches ASR ordering.

Expected (if C4 holds):
    TranAD (low norm, low ASR) < LSTM-AE (mid) < USAD (high norm, high ASR)

If ordering matches → gradient norm IS architectural → C4 stands
If ordering doesn't match → gradient norm is dataset-driven → C4 falls
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import create_model
from data_loader import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DETECTORS = ["lstm_ae", "tranad", "usad"]
DATASET = "swat"
N_GRAD_SAMPLES = 2000
SEEDS = [42, 123, 456]

# Real ASR values (AutoAttack eps=0.05 unconstrained, from aggregates)
KNOWN_ASR = {
    "lstm_ae": 0.304,
    "tranad": 0.255,
    "usad": 0.509,
}


def infer_window_size_from_checkpoint(model_path: str, det_name: str, n_features: int) -> int:
    """Infer window_size from saved checkpoint dimensions."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    if det_name == "lstm_ae":
        # LSTM processes sequentially — window_size doesn't affect weight shapes.
        # Check positional encoding or just look at other model files for consistency.
        # Fallback: scan for any layer where dim > n_features
        return None
    elif det_name == "usad":
        # encoder first layer: [hidden, window*features]
        first_key = [k for k in state.keys() if "encoder" in k and "weight" in k][0]
        input_dim = state[first_key].shape[1]
        window = input_dim // n_features
        return window
    elif det_name == "tranad":
        # pos_encoding shape = [1, window, hidden]
        for k, v in state.items():
            if "pos_encoding" in k or "pos_enc" in k:
                return v.shape[1]
        # Fallback: check input projection
        for k, v in state.items():
            if "input" in k.lower() and "weight" in k.lower():
                input_dim = v.shape[1]
                if input_dim > n_features:
                    return input_dim // n_features
        return None

    return None


def compute_gradient_norms(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 1000,
    attack_only: bool = False,
) -> np.ndarray:
    """Compute gradient L2 norms of anomaly score w.r.t. input."""
    model.train()
    model.to(device)

    grad_norms = []
    count = 0

    for batch_x, batch_y in data_loader:
        if count >= n_samples:
            break

        if attack_only:
            mask = batch_y == 1
            if not mask.any():
                continue
            batch_x = batch_x[mask]

        batch_x = batch_x.to(device).requires_grad_(True)

        try:
            scores = model.anomaly_score(batch_x)
        except Exception as e:
            logger.warning(f"anomaly_score failed: {e}")
            batch_x.requires_grad_(False)
            continue

        # Sum and backward for efficiency
        loss = scores.sum()
        loss.backward()

        if batch_x.grad is not None:
            norms = torch.norm(
                batch_x.grad.reshape(len(batch_x), -1), dim=1
            )
            for n in norms.detach().cpu().numpy():
                if count >= n_samples:
                    break
                grad_norms.append(float(n))
                count += 1

        batch_x.requires_grad_(False)
        model.zero_grad()

    return np.array(grad_norms) if grad_norms else np.array([0.0])


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Experiment A: Within-Dataset Gradient Norm on {DATASET}")
    logger.info(f"Detectors: {DETECTORS}")
    logger.info(f"Seeds: {SEEDS}")

    results_dir = Path(config["output"]["results_dir"])
    models_dir = Path(config["output"]["models_dir"])
    data_dir = config["data"][f"{DATASET}_dir"]
    default_window = config["data"]["window_size"]
    n_features_cache = {}

    all_results = {}

    for det_name in DETECTORS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Detector: {det_name}")

        seed_grad_norms = []

        for seed in SEEDS:
            # Find model checkpoint
            model_path = models_dir / f"{det_name}_{DATASET}_seed{seed}_best.pt"
            if not model_path.exists():
                model_path = models_dir / f"{det_name}_{DATASET}_best.pt"
            if not model_path.exists():
                logger.warning(f"  Seed {seed}: model not found, skipping")
                continue

            # Infer window size from checkpoint
            if DATASET not in n_features_cache:
                # Quick load to get n_features
                tmp_loader, _, _, meta = get_dataloaders(
                    DATASET, data_dir,
                    window_size=default_window, stride=1,
                    batch_size=2, val_ratio=config["data"]["val_ratio"],
                )
                n_features_cache[DATASET] = meta["n_features"]

            n_features = n_features_cache[DATASET]
            inferred_window = infer_window_size_from_checkpoint(
                str(model_path), det_name, n_features
            )
            window_size = inferred_window if inferred_window else default_window
            logger.info(f"  Seed {seed}: window_size={window_size} (inferred={inferred_window})")

            # Load data with correct window size
            _, _, test_loader, metadata = get_dataloaders(
                DATASET, data_dir,
                window_size=window_size,
                stride=1,
                batch_size=64,
                val_ratio=config["data"]["val_ratio"],
            )

            # Create and load model
            model = create_model(
                det_name, n_features,
                window_size=window_size,
                hidden_dim=config["detectors"]["common"]["hidden_dim"],
            )

            try:
                model.load_state_dict(
                    torch.load(str(model_path), map_location=device, weights_only=True)
                )
            except RuntimeError as e:
                logger.error(f"  Seed {seed}: load failed — {e}")
                # Try with strict=False to see what's happening
                try:
                    model.load_state_dict(
                        torch.load(str(model_path), map_location=device, weights_only=True),
                        strict=False,
                    )
                    logger.warning(f"  Seed {seed}: loaded with strict=False")
                except Exception:
                    logger.error(f"  Seed {seed}: completely failed, skipping")
                    continue

            # Compute gradient norms on attack windows
            logger.info(f"  Seed {seed}: computing gradient norms ({N_GRAD_SAMPLES} samples)...")

            # All test data
            all_norms = compute_gradient_norms(
                model, test_loader, device, n_samples=N_GRAD_SAMPLES, attack_only=False
            )
            # Attack windows only
            attack_norms = compute_gradient_norms(
                model, test_loader, device, n_samples=N_GRAD_SAMPLES, attack_only=True
            )

            logger.info(
                f"  Seed {seed}: all_mean={np.mean(all_norms):.6f} "
                f"attack_mean={np.mean(attack_norms):.6f} "
                f"n_all={len(all_norms)} n_attack={len(attack_norms)}"
            )

            seed_grad_norms.append({
                "seed": seed,
                "all_mean": float(np.mean(all_norms)),
                "all_std": float(np.std(all_norms)),
                "all_median": float(np.median(all_norms)),
                "attack_mean": float(np.mean(attack_norms)),
                "attack_std": float(np.std(attack_norms)),
                "n_all": len(all_norms),
                "n_attack": len(attack_norms),
            })

        if not seed_grad_norms:
            logger.error(f"  No results for {det_name}, skipping")
            continue

        # Aggregate across seeds
        all_means = [s["all_mean"] for s in seed_grad_norms]
        attack_means = [s["attack_mean"] for s in seed_grad_norms]

        all_results[det_name] = {
            "detector": det_name,
            "dataset": DATASET,
            "grad_norm_all_mean": float(np.mean(all_means)),
            "grad_norm_all_std": float(np.std(all_means)),
            "grad_norm_attack_mean": float(np.mean(attack_means)),
            "grad_norm_attack_std": float(np.std(attack_means)),
            "known_asr": KNOWN_ASR.get(det_name, None),
            "per_seed": seed_grad_norms,
        }

    # Save
    out_path = results_dir / "exp_a_gradient_within_swat.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    # === VERDICT ===
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT A: WITHIN-DATASET GRADIENT NORM VERDICT")
    logger.info(f"{'='*60}\n")

    if len(all_results) < 2:
        logger.error("Not enough detectors succeeded. Cannot compute correlation.")
        return

    names = []
    gnorms = []
    asrs = []

    logger.info(f"{'Detector':10s} {'Grad Norm':>12s} {'ASR':>8s}")
    logger.info("-" * 32)
    for det in DETECTORS:
        if det in all_results:
            gn = all_results[det]["grad_norm_attack_mean"]
            asr = KNOWN_ASR.get(det, 0)
            logger.info(f"{det:10s} {gn:12.6f} {asr:8.3f}")
            names.append(det)
            gnorms.append(gn)
            asrs.append(asr)

    # Check ordering
    gn_order = sorted(range(len(gnorms)), key=lambda i: gnorms[i])
    asr_order = sorted(range(len(asrs)), key=lambda i: asrs[i])

    gn_ranking = [names[i] for i in gn_order]
    asr_ranking = [names[i] for i in asr_order]

    logger.info(f"\nGradient norm ranking (low→high): {' < '.join(gn_ranking)}")
    logger.info(f"ASR ranking (low→high):           {' < '.join(asr_ranking)}")

    if gn_ranking == asr_ranking:
        logger.info("\n✅ RANKINGS MATCH → gradient norm IS architectural")
        logger.info("   C4 contribution STANDS")
    else:
        logger.info("\n❌ RANKINGS DO NOT MATCH → gradient norm may be confounded")
        logger.info("   C4 contribution NEEDS REVISION")

    # Correlation (n=3, so low power, but ordering matters more)
    if len(gnorms) >= 3:
        from scipy import stats

        r, p = stats.pearsonr(gnorms, asrs)
        rho, p_rho = stats.spearmanr(gnorms, asrs)
        logger.info(f"\nPearson r = {r:.4f} (p = {p:.4f})")
        logger.info(f"Spearman rho = {rho:.4f} (p = {p_rho:.4f})")
        logger.info(f"R² = {r**2:.4f}")

    logger.info("\nEXP_A_DONE")


if __name__ == "__main__":
    main()
