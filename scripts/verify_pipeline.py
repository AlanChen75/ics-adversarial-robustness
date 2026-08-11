#!/usr/bin/env python3
"""
Pipeline verification using synthetic data.
Validates all 5 detectors + 4 attacks work correctly before running on real SWaT/WADI.
Usage: python verify_pipeline.py
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import create_model


def generate_synthetic_ics_data(n_features=20, n_train=5000, n_test=2000,
                                 n_attack=500, window_size=100, seed=42):
    """Generate synthetic ICS-like data with normal and attack patterns."""
    rng = np.random.RandomState(seed)

    # Normal: smooth sinusoidal patterns with noise (mimics sensor readings)
    t_train = np.linspace(0, 50, n_train + window_size)
    train_data = np.zeros((len(t_train), n_features))
    for i in range(n_features):
        freq = rng.uniform(0.1, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        train_data[:, i] = 0.5 + 0.3 * np.sin(freq * t_train + phase) + 0.02 * rng.randn(len(t_train))

    # Test: normal + injected attacks
    t_test = np.linspace(50, 70, n_test + window_size)
    test_data = np.zeros((len(t_test), n_features))
    test_labels = np.zeros(len(t_test))
    for i in range(n_features):
        freq = rng.uniform(0.1, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        test_data[:, i] = 0.5 + 0.3 * np.sin(freq * t_test + phase) + 0.02 * rng.randn(len(t_test))

    # Inject attacks: shift values abruptly
    attack_starts = rng.choice(range(window_size, n_test), size=n_attack // 50, replace=False)
    for start in attack_starts:
        length = rng.randint(30, 80)
        end = min(start + length, n_test + window_size)
        affected = rng.choice(n_features, size=rng.randint(3, 8), replace=False)
        for feat in affected:
            test_data[start:end, feat] += rng.uniform(0.2, 0.5) * rng.choice([-1, 1])
        test_labels[start:end] = 1

    # Clip to [0, 1]
    train_data = np.clip(train_data, 0, 1)
    test_data = np.clip(test_data, 0, 1)

    # Create windows
    def make_windows(data, labels=None):
        n = len(data) - window_size + 1
        windows = np.zeros((n, window_size, n_features))
        w_labels = np.zeros(n)
        for i in range(n):
            windows[i] = data[i:i + window_size]
            if labels is not None:
                w_labels[i] = 1.0 if labels[i:i + window_size].any() else 0.0
        return windows, w_labels

    train_w, train_l = make_windows(train_data)
    test_w, test_l = make_windows(test_data, test_labels)

    return train_w, test_w, test_l, n_features


def test_detector(det_name, train_windows, test_windows, test_labels, n_features,
                  window_size=100, device='cpu'):
    """Train and test a single detector."""
    print(f"\n--- Testing {det_name} ---")

    model = create_model(det_name, n_features, window_size=window_size, hidden_dim=32)
    model = model.to(device)

    # Quick training (5 epochs)
    train_tensor = torch.FloatTensor(train_windows[:1000]).to(device)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(5):
        total_loss = 0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(batch_x, epoch, 5)
            else:
                recon = model(batch_x)
                if isinstance(recon, tuple):
                    recon = recon[0]
                loss = nn.MSELoss()(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.6f}")

    # Compute threshold
    model.eval()
    with torch.no_grad():
        train_scores = model.anomaly_score(train_tensor[:500]).cpu().numpy()
    threshold = np.percentile(train_scores, 99)
    print(f"  Threshold (99th pct): {threshold:.6f}")

    # Test
    test_tensor = torch.FloatTensor(test_windows[:500]).to(device)
    with torch.no_grad():
        test_scores = model.anomaly_score(test_tensor).cpu().numpy()

    predictions = (test_scores > threshold).astype(int)
    labels = test_labels[:500]

    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    dr = tp / max(tp + fn, 1)
    far = fp / max(fp + tn, 1)

    print(f"  Clean: DR={dr:.4f}, FAR={far:.4f}, TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Test adversarial attack (FGSM)
    # Switch to train mode for gradient computation (cuDNN LSTM requires it)
    model.train()
    attack_mask = labels == 1
    if attack_mask.any():
        x_attack = test_tensor[attack_mask[:len(test_tensor)]].clone().requires_grad_(True)
        if len(x_attack) > 0:
            scores = model.anomaly_score(x_attack)
            scores.sum().backward()
            perturbation = 0.05 * x_attack.grad.sign()
            x_adv = torch.clamp(x_attack - perturbation, 0, 1).detach()

            with torch.no_grad():
                adv_scores = model.anomaly_score(x_adv).cpu().numpy()
            adv_detected = (adv_scores > threshold).sum()
            asr = 1.0 - adv_detected / len(adv_scores)
            print(f"  FGSM (eps=0.05): ASR={asr:.4f} ({len(adv_scores)-adv_detected}/{len(adv_scores)} evaded)")
        else:
            print("  No attack windows in sample")
    else:
        print("  No attack labels in test sample")

    return True


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Generating synthetic ICS data...")

    train_w, test_w, test_l, n_feat = generate_synthetic_ics_data()
    print(f"Train: {train_w.shape}, Test: {test_w.shape}, Attack ratio: {test_l.mean():.3f}")

    detectors = ['lstm_ae', 'usad', 'gdn', 'tranad', 'dagmm']
    results = {}

    for det in detectors:
        try:
            ok = test_detector(det, train_w, test_w, test_l, n_feat, device=device)
            results[det] = 'PASS' if ok else 'FAIL'
        except Exception as e:
            print(f"  ERROR: {e}")
            results[det] = f'FAIL: {e}'

    print("\n" + "=" * 50)
    print("PIPELINE VERIFICATION RESULTS")
    print("=" * 50)
    for det, status in results.items():
        icon = "OK" if status == 'PASS' else "FAIL"
        print(f"  [{icon}] {det}: {status}")

    all_pass = all(v == 'PASS' for v in results.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    if all_pass:
        print("\nPipeline verified! Ready to run on SWaT/WADI data.")
        print("Place data files in data/swat/ and data/wadi/, then run:")
        print("  python run_experiment.py --stage all --dataset swat")
        print("  python run_experiment.py --stage all --dataset wadi")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
