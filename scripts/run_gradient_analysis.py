#!/usr/bin/env python3
"""
Gradient Norm Analysis — Robustness Predictor for ICS Anomaly Detectors.

Hypothesis: detectors with smoother loss landscapes (lower gradient norms)
are harder to attack. If gradient norm correlates with ASR, it becomes
a cheap proxy for robustness assessment without running full attacks.

Metrics computed per detector:
  1. Mean gradient norm (L2) of anomaly score w.r.t. input
  2. Max gradient norm
  3. Gradient norm variance (stability)
  4. Jacobian spectral norm (largest singular value)

Then correlate with known ASR from main experiments.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def compute_gradient_norms(model, data_loader, device, n_samples=1000):
    """Compute gradient norms of anomaly score w.r.t. input."""
    model.train()  # need gradients, train mode for LSTM backward
    model.to(device)

    grad_norms = []
    count = 0

    for batch_x, batch_y in data_loader:
        if count >= n_samples:
            break

        batch_x = batch_x.to(device).requires_grad_(True)
        scores = model.anomaly_score(batch_x)

        # Compute gradient for each sample
        for i in range(len(scores)):
            if count >= n_samples:
                break

            model.zero_grad()
            if batch_x.grad is not None:
                batch_x.grad.zero_()

            scores[i].backward(retain_graph=True)

            if batch_x.grad is not None:
                grad = batch_x.grad[i].detach().cpu()
                norm = torch.norm(grad).item()
                grad_norms.append(norm)

            count += 1

        # Detach to free graph
        scores.detach()
        batch_x.requires_grad_(False)

    return np.array(grad_norms)


def compute_jacobian_spectral_norm(model, data_loader, device, n_samples=100):
    """Compute spectral norm of Jacobian (largest singular value).
    More expensive but captures directional sensitivity."""
    model.train()
    model.to(device)

    spectral_norms = []
    count = 0

    for batch_x, _ in data_loader:
        if count >= n_samples:
            break

        # Process one sample at a time (Jacobian is expensive)
        for i in range(min(len(batch_x), n_samples - count)):
            x = batch_x[i:i+1].to(device).requires_grad_(True)
            score = model.anomaly_score(x)

            # Compute Jacobian row (score is scalar, so Jacobian = gradient)
            score.backward()
            if x.grad is not None:
                grad = x.grad[0].detach().cpu().reshape(-1)
                # For scalar output, spectral norm = L2 norm of gradient
                spectral_norms.append(torch.norm(grad).item())

            count += 1

    return np.array(spectral_norms)


def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    results_dir = Path(config['output']['results_dir'])
    models_dir = Path(config['output']['models_dir'])

    all_results = {}

    for dataset_name in ['swat', 'wadi']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Dataset: {dataset_name}")

        data_dir = config['data'][f'{dataset_name}_dir']

        # Load data
        train_loader, val_loader, test_loader, metadata = get_dataloaders(
            dataset_name, data_dir,
            window_size=config['data']['window_size'],
            stride=config['data'].get('stride', 1),
            batch_size=64,  # smaller batch for gradient computation
            val_ratio=config['data']['val_ratio'],
        )
        n_features = metadata['n_features']

        for det_name in config['detectors']['architectures']:
            logger.info(f"\n  Detector: {det_name}")

            model = create_model(det_name, n_features,
                               window_size=config['data']['window_size'],
                               hidden_dim=config['detectors']['common']['hidden_dim'])

            # Load model (seed 42)
            model_path = models_dir / f"{det_name}_{dataset_name}_seed42_best.pt"
            if not model_path.exists():
                model_path = models_dir / f"{det_name}_{dataset_name}_best.pt"
            if not model_path.exists():
                logger.warning(f"    Model not found, skipping")
                continue

            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

            # Compute gradient norms on TEST data (attack samples)
            logger.info(f"    Computing gradient norms (1000 samples)...")
            grad_norms = compute_gradient_norms(model, test_loader, device, n_samples=1000)

            # Compute on ATTACK windows only
            attack_grad_norms = []
            model.train()
            count = 0
            for batch_x, batch_y in test_loader:
                if count >= 500:
                    break
                attack_mask = batch_y == 1
                if not attack_mask.any():
                    continue
                x_attack = batch_x[attack_mask].to(device).requires_grad_(True)
                scores = model.anomaly_score(x_attack)
                scores.sum().backward()
                if x_attack.grad is not None:
                    norms = torch.norm(x_attack.grad.reshape(len(x_attack), -1), dim=1)
                    attack_grad_norms.extend(norms.detach().cpu().numpy().tolist())
                    count += len(norms)
                x_attack.requires_grad_(False)

            attack_grad_norms = np.array(attack_grad_norms) if attack_grad_norms else np.array([0.0])

            result = {
                'detector': det_name,
                'dataset': dataset_name,
                'n_samples': len(grad_norms),
                'grad_norm_mean': float(np.mean(grad_norms)),
                'grad_norm_std': float(np.std(grad_norms)),
                'grad_norm_median': float(np.median(grad_norms)),
                'grad_norm_max': float(np.max(grad_norms)),
                'grad_norm_p95': float(np.percentile(grad_norms, 95)),
                'attack_grad_norm_mean': float(np.mean(attack_grad_norms)),
                'attack_grad_norm_std': float(np.std(attack_grad_norms)),
                'n_attack_samples': len(attack_grad_norms),
            }

            key = f"{det_name}_{dataset_name}"
            all_results[key] = result
            logger.info(f"    grad_norm: mean={result['grad_norm_mean']:.4f} "
                       f"std={result['grad_norm_std']:.4f} "
                       f"attack_mean={result['attack_grad_norm_mean']:.4f}")

    # Save results
    out_path = results_dir / 'gradient_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    # Print correlation summary
    logger.info(f"\n{'='*60}")
    logger.info("GRADIENT NORM vs ASR CORRELATION SUMMARY")
    logger.info(f"{'='*60}")

    # Known ASR values from main experiments (AutoAttack eps=0.05)
    known_asr = {
        'lstm_ae_swat': 0.321,
        'usad_swat': 0.892,
        'tranad_swat': 0.263,
        'gdn_swat': 0.999,
        'dagmm_swat': 0.997,
        'lstm_ae_wadi': 0.517,
        'usad_wadi': 0.601,
        'tranad_wadi': 0.532,
        'gdn_wadi': 0.803,
        'dagmm_wadi': 0.963,
    }

    grad_means = []
    asr_values = []
    labels = []

    logger.info(f"{'Detector':20s} {'Grad Norm':>12s} {'ASR':>8s}")
    logger.info("-" * 42)
    for key, result in sorted(all_results.items()):
        gn = result['attack_grad_norm_mean']
        asr = known_asr.get(key, None)
        if asr is not None:
            grad_means.append(gn)
            asr_values.append(asr)
            labels.append(key)
            logger.info(f"{key:20s} {gn:12.4f} {asr:8.3f}")

    if len(grad_means) >= 3:
        from scipy import stats
        r, p = stats.pearsonr(grad_means, asr_values)
        rho, p_rho = stats.spearmanr(grad_means, asr_values)
        logger.info(f"\nPearson r = {r:.4f} (p = {p:.4f})")
        logger.info(f"Spearman rho = {rho:.4f} (p = {p_rho:.4f})")

        # R-squared
        r_sq = r ** 2
        logger.info(f"R² = {r_sq:.4f}")

        if abs(r) > 0.7:
            logger.info("*** STRONG CORRELATION: gradient norm is a viable robustness predictor ***")
        elif abs(r) > 0.4:
            logger.info("** MODERATE CORRELATION: gradient norm has some predictive value **")
        else:
            logger.info("* WEAK CORRELATION: gradient norm alone is insufficient *")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
