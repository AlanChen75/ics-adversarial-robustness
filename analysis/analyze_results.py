#!/usr/bin/env python3
"""
ICS Adversarial Robustness — Complete Figure & Table Generator
Reads all experiment result JSONs and produces publication-ready figures.

Usage:
    python analyze_results.py [--results-dir PATH] [--figures-dir PATH]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Global academic style — must be applied before any figure is created
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "svg.fonttype": "none",
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "supplementary"
FIGURES_DIR = REPO_ROOT / "figures"

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
DETECTOR_ORDER = ["dagmm", "gdn", "lstm_ae", "tranad", "usad"]
DETECTOR_LABELS = {
    "dagmm": "DAGMM",
    "gdn": "GDN",
    "lstm_ae": "LSTM-AE",
    "tranad": "TranAD",
    "usad": "USAD",
}
ATTACK_ORDER = ["fgsm", "ifgsm", "pgd", "cw", "autoattack"]
ATTACK_LABELS = {
    "fgsm": "FGSM",
    "ifgsm": "I-FGSM",
    "pgd": "PGD",
    "cw": "C&W",
    "autoattack": "AutoAttack",
}
DATASET_LABELS = {"swat": "SWaT", "wadi": "WADI", "smd": "SMD"}
EPSILON_ORDER = ["0.01", "0.05", "0.1"]

# Fixed colors per detector (muted, academic palette)
DETECTOR_COLORS = {
    "dagmm":   "#999999",   # grey (broken detector)
    "gdn":     "#AAAAAA",   # light grey (broken)
    "lstm_ae": "#2166AC",   # muted blue
    "tranad":  "#1B7837",   # muted green
    "usad":    "#B2182B",   # muted red
}

# Bar chart colors (undefended vs defended)
BAR_UNDEF = "#D6604D"   # muted red
BAR_DEF   = "#4393C3"   # muted blue

# Font sizes
FS_TITLE  = 14
FS_LABEL  = 13
FS_TICK   = 11
FS_ANNOT  = 10
FS_LEGEND = 11

STYLE = "seaborn-v0_8-whitegrid"  # kept for fallback reference only; not used


def _apply_academic_style(ax):
    """Apply white background, full frame border, outward ticks."""
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    # Journal figure style: full border around the plot area.
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(direction="out", length=4, width=0.8)

# ---------------------------------------------------------------------------
# File-name parser
# ---------------------------------------------------------------------------
_ATTACK_PAT = re.compile(
    r"attack_(?P<detector>\w+)_(?P<dataset>\w+)_(?P<attack>\w+)"
    r"_(?P<epsilon>[\d.]+|None)_(?P<constraint>\w+)_aggregate\.json$"
)
_BASELINE_PAT = re.compile(
    r"baseline_(?P<detector>\w+)_(?P<dataset>\w+)"
    r"(?:_(?P<variant>\w+))?_aggregate\.json$"
)


def _load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _safe_mean(d: dict, key: str) -> float:
    """Return mean of a metric dict, or NaN if key absent."""
    if key not in d:
        return float("nan")
    val = d[key]
    if isinstance(val, dict):
        return val.get("mean", float("nan"))
    return float(val)


def _safe_std(d: dict, key: str) -> float:
    if key not in d:
        return float("nan")
    val = d[key]
    if isinstance(val, dict):
        return val.get("std", 0.0)
    return 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_attack_results(results_dir: Path) -> list[dict]:
    """
    Load all attack_*_aggregate.json files into a list of flat dicts.
    Each dict: detector, dataset, attack, epsilon, constraint,
               asr_mean, asr_std, dr_mean, dr_std, f1_mean, f1_std,
               far_mean, far_std
    """
    rows = []
    for path in sorted(results_dir.glob("attack_*_aggregate.json")):
        m = _ATTACK_PAT.match(path.name)
        if not m:
            continue
        try:
            d = _load_json(path)
        except Exception as exc:
            print(f"  [WARN] Cannot read {path.name}: {exc}", file=sys.stderr)
            continue
        rows.append({
            "detector":   m.group("detector"),
            "dataset":    m.group("dataset"),
            "attack":     m.group("attack"),
            "epsilon":    m.group("epsilon"),
            "constraint": m.group("constraint"),
            "asr_mean":   _safe_mean(d, "attack_success_rate"),
            "asr_std":    _safe_std(d,  "attack_success_rate"),
            "dr_mean":    _safe_mean(d, "detection_rate"),
            "dr_std":     _safe_std(d,  "detection_rate"),
            "f1_mean":    _safe_mean(d, "f1_score"),
            "f1_std":     _safe_std(d,  "f1_score"),
            "far_mean":   _safe_mean(d, "false_alarm_rate"),
            "far_std":    _safe_std(d,  "false_alarm_rate"),
        })
    print(f"Loaded {len(rows)} attack result records.")
    return rows


def load_baseline_results(results_dir: Path) -> dict:
    """
    Return dict keyed by (detector, dataset, variant) where variant is
    '' (normal) or 'defended'.
    """
    baselines = {}
    for path in sorted(results_dir.glob("baseline_*_aggregate.json")):
        # skip per-seed files
        if re.search(r"seed\d+", path.name):
            continue
        m = _BASELINE_PAT.match(path.name)
        if not m:
            continue
        detector = m.group("detector")
        dataset  = m.group("dataset")
        variant  = m.group("variant") or ""
        # skip variants that aren't '' or 'defended'
        if variant not in ("", "defended"):
            continue
        try:
            d = _load_json(path)
        except Exception as exc:
            print(f"  [WARN] Cannot read {path.name}: {exc}", file=sys.stderr)
            continue
        baselines[(detector, dataset, variant)] = {
            "f1_mean":  _safe_mean(d, "f1_score"),
            "f1_std":   _safe_std(d,  "f1_score"),
            "dr_mean":  _safe_mean(d, "detection_rate"),
            "dr_std":   _safe_std(d,  "detection_rate"),
            "far_mean": _safe_mean(d, "false_alarm_rate"),
            "far_std":  _safe_std(d,  "false_alarm_rate"),
        }
    print(f"Loaded {len(baselines)} baseline records.")
    return baselines


def load_gradient_norms(results_dir: Path) -> dict:
    """Legacy loader kept for statistics printer compatibility; returns empty dict."""
    return {}


def load_threshold_sweep(results_dir: Path, dataset: str) -> dict:
    path = results_dir / f"threshold_sweep_{dataset}.json"
    if not path.exists():
        print(f"  [WARN] {path.name} not found", file=sys.stderr)
        return {}
    return _load_json(path)


# ---------------------------------------------------------------------------
# Helpers to query loaded data
# ---------------------------------------------------------------------------

def filter_rows(rows, **kwargs) -> list[dict]:
    result = rows
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            result = [r for r in result if r[k] in v]
        else:
            result = [r for r in result if r[k] == v]
    return result


def pivot_heatmap(rows, detector_order, attack_order, metric="asr_mean") -> np.ndarray:
    """Build (n_detectors × n_attacks) matrix, NaN where missing."""
    matrix = np.full((len(detector_order), len(attack_order)), np.nan)
    for r in rows:
        if r["detector"] in detector_order and r["attack"] in attack_order:
            i = detector_order.index(r["detector"])
            j = attack_order.index(r["attack"])
            matrix[i, j] = r[metric]
    return matrix


# ---------------------------------------------------------------------------
# Figure 2 & 3: ASR Heatmaps
# ---------------------------------------------------------------------------

SMD_DETECTOR_ORDER = ["lstm_ae", "tranad", "usad"]


def plot_asr_heatmap(rows, dataset: str, figures_dir: Path):
    fig_num = 2 if dataset == "swat" else 3
    filename = f"fig{fig_num}_heatmap_{dataset}.png"

    # SMD has only 3 detectors; adjust accordingly
    det_order = DETECTOR_ORDER if dataset != "smd" else SMD_DETECTOR_ORDER

    # For most attacks use eps=0.05; C&W uses eps=None (L2 attack, no eps bound)
    subset_eps = filter_rows(rows,
                             dataset=dataset,
                             epsilon="0.05",
                             constraint="unconstrained")
    subset_cw = filter_rows(rows,
                            dataset=dataset,
                            attack="cw",
                            epsilon="None",
                            constraint="unconstrained")
    subset = subset_eps + [r for r in subset_cw if r not in subset_eps]
    matrix = pivot_heatmap(subset, det_order, ATTACK_ORDER)

    # SMD ASR is very low — use a tight colormap (0–0.1 max)
    vmax = 0.1 if dataset == "smd" else 1.0
    cmap = "YlOrRd" if dataset == "smd" else "RdYlGn_r"

    # Ensure minimum row height for SMD (3 rows) to avoid squished heatmap
    fig_h = max(4.5, len(det_order) * 1.4)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(ATTACK_ORDER)))
    ax.set_xticklabels([ATTACK_LABELS[a] for a in ATTACK_ORDER],
                       fontsize=FS_TICK)
    ax.set_yticks(range(len(det_order)))
    ax.set_yticklabels([DETECTOR_LABELS[d] for d in det_order],
                       fontsize=FS_TICK)

    # Remove grid background from heatmap axes
    ax.grid(False)

    # Annotate each cell — SMD: show as percentage; cell font 11pt
    for i in range(len(det_order)):
        for j in range(len(ATTACK_ORDER)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "grey"
            elif dataset == "smd":
                # Show as percentage for SMD (very low values)
                text = f"{val*100:.1f}%"
                color = "black"  # all cells are light on SMD's 0–10% scale
            else:
                text = f"{val:.2f}"
                # white text on dark cells, black on light
                color = "white" if (val > 0.7 or val < 0.3) else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=20, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # fig3: remove "max=10%" from colorbar label
    cbar_label = "Attack Success Rate (ASR)"
    cbar.set_label(cbar_label, fontsize=FS_LABEL)
    if dataset == "smd":
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%")
        )
    cbar.ax.tick_params(labelsize=FS_TICK)

    ax.set_title(
        f"Attack Success Rate — {DATASET_LABELS[dataset]} (ε = 0.05, unconstrained)",
        fontsize=FS_TITLE, fontweight="bold", pad=12
    )
    ax.set_xlabel("Attack Method", fontsize=FS_LABEL, labelpad=8)
    ax.set_ylabel("Anomaly Detector", fontsize=FS_LABEL, labelpad=8)

    fig.tight_layout()
    out = figures_dir / filename
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out).replace(".png", ".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: DR vs ε under PGD on SWaT
# ---------------------------------------------------------------------------

def plot_epsilon_dr(rows, figures_dir: Path):
    subset = filter_rows(rows, dataset="swat", attack="pgd",
                         constraint="unconstrained")
    eps_vals = [float(e) for e in EPSILON_ORDER]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_academic_style(ax)

    # 3 main detectors: solid lines with DET_COLORS
    # GDN/DAGMM: thin dashed grey lines
    BROKEN_DETS = {"dagmm", "gdn"}

    plotted = []
    for det in DETECTOR_ORDER:
        dr_means, dr_stds = [], []
        for eps_str in EPSILON_ORDER:
            matching = filter_rows(subset, detector=det, epsilon=eps_str)
            if matching:
                dr_means.append(matching[0]["dr_mean"])
                dr_stds.append(matching[0]["dr_std"])
            else:
                dr_means.append(np.nan)
                dr_stds.append(0.0)

        dr_arr = np.array(dr_means)
        std_arr = np.array(dr_stds)

        # Only plot if we have at least 2 valid points
        valid = ~np.isnan(dr_arr)
        if valid.sum() < 2:
            continue

        color = DETECTOR_COLORS[det]
        label = DETECTOR_LABELS[det]
        is_broken = det in BROKEN_DETS
        ax.errorbar(
            [eps_vals[k] for k in range(len(eps_vals)) if valid[k]],
            dr_arr[valid],
            yerr=std_arr[valid],
            label=label,
            color=color,
            marker="o",
            markersize=6 if is_broken else 8,
            linewidth=1.0 if is_broken else 2.0,
            linestyle="--" if is_broken else "-",
            capsize=4,
            capthick=1.0 if is_broken else 1.5,
            alpha=0.6 if is_broken else 1.0,
        )
        plotted.append(det)

    ax.set_xlabel("Perturbation Budget ε", fontsize=FS_LABEL)
    ax.set_ylabel("Detection Rate (DR)", fontsize=FS_LABEL)
    ax.set_title(
        "Detection Rate vs. ε Under PGD Attack (SWaT, unconstrained)",
        fontsize=FS_TITLE, fontweight="bold"
    )
    ax.set_xticks(eps_vals)
    ax.set_xticklabels([str(e) for e in eps_vals], fontsize=FS_TICK)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylim(bottom=0)
    # Journal figure style: legend inside the plot area.
    ax.legend(fontsize=FS_LEGEND, loc="best",
              frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()

    out = figures_dir / "fig4_epsilon_dr_swat.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out).replace(".png", ".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5: Defense Comparison (undefended vs defended ASR)
# ---------------------------------------------------------------------------

def plot_defense_comparison(rows, figures_dir: Path):
    # Use ε=0.05, attacks: fgsm, pgd; SWaT (5 detectors) and SMD (3 detectors)
    attacks_shown = ["fgsm", "pgd"]
    datasets_shown = ["swat", "smd"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.set_facecolor("white")

    bar_width = 0.35

    for ax, dataset in zip(axes, datasets_shown):
        _apply_academic_style(ax)
        det_order = DETECTOR_ORDER if dataset != "smd" else SMD_DETECTOR_ORDER
        x = np.arange(len(det_order))
        # Average ASR across the shown attacks
        undef_means, undef_stds = [], []
        def_means,   def_stds   = [], []

        for det in det_order:
            # Undefended: constraint='unconstrained'
            u_vals = filter_rows(rows, detector=det, dataset=dataset,
                                 epsilon="0.05", constraint="unconstrained",
                                 attack=attacks_shown)
            # Defended
            d_vals = filter_rows(rows, detector=det, dataset=dataset,
                                 constraint="defended",
                                 attack=attacks_shown)

            def avg_metric(rlist, key):
                vals = [r[key] for r in rlist if not np.isnan(r[key])]
                return (np.mean(vals), np.std(vals)) if vals else (np.nan, 0)

            um, us = avg_metric(u_vals, "asr_mean")
            dm, ds = avg_metric(d_vals, "asr_mean")
            undef_means.append(um)
            undef_stds.append(us)
            def_means.append(dm)
            def_stds.append(ds)

        bars1 = ax.bar(x - bar_width / 2, undef_means, bar_width,
                       yerr=undef_stds, capsize=4,
                       label="Undefended", color=BAR_UNDEF, alpha=0.9,
                       error_kw={"elinewidth": 1.5})
        bars2 = ax.bar(x + bar_width / 2, def_means, bar_width,
                       yerr=def_stds, capsize=4,
                       label="AT-Defended", color=BAR_DEF, alpha=0.9,
                       error_kw={"elinewidth": 1.5})

        # Annotate all bars
        for rect, val in zip(list(bars1) + list(bars2),
                             undef_means + def_means):
            if not np.isnan(val):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.015,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=FS_ANNOT - 1, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([DETECTOR_LABELS[d] for d in det_order],
                           fontsize=FS_TICK)
        ax.set_title(DATASET_LABELS[dataset], fontsize=FS_TITLE,
                     fontweight="bold")
        ax.set_xlabel("Anomaly Detector", fontsize=FS_LABEL)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND, frameon=True, edgecolor="black", fancybox=False)

    axes[0].set_ylabel("Attack Success Rate (ASR)", fontsize=FS_LABEL)
    fig.suptitle(
        "Adversarial Training Defense Effectiveness",
        fontsize=FS_TITLE, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    out = figures_dir / "fig5_defense_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out).replace(".png", ".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 6: Constraint Effect (unconstrained vs constrained ASR)
# ---------------------------------------------------------------------------

def plot_constraint_effect(rows, figures_dir: Path):
    # Only PGD has constrained variants; use ε=0.05, SWaT + SMD
    datasets_shown = ["swat", "smd"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.set_facecolor("white")

    bar_width = 0.35

    for ax, dataset in zip(axes, datasets_shown):
        _apply_academic_style(ax)
        det_order = DETECTOR_ORDER if dataset != "smd" else SMD_DETECTOR_ORDER
        x = np.arange(len(det_order))
        unc_means, unc_stds = [], []
        con_means, con_stds = [], []

        for det in det_order:
            unc = filter_rows(rows, detector=det, dataset=dataset,
                              attack="pgd", epsilon="0.05",
                              constraint="unconstrained")
            con = filter_rows(rows, detector=det, dataset=dataset,
                              attack="pgd", epsilon="0.05",
                              constraint="constrained")

            um = unc[0]["asr_mean"] if unc else np.nan
            us = unc[0]["asr_std"]  if unc else 0.0
            cm = con[0]["asr_mean"] if con else np.nan
            cs = con[0]["asr_std"]  if con else 0.0

            unc_means.append(um)
            unc_stds.append(us)
            con_means.append(cm)
            con_stds.append(cs)

        bars1 = ax.bar(x - bar_width / 2, unc_means, bar_width,
                       yerr=unc_stds, capsize=4,
                       label="Unconstrained", color=BAR_UNDEF, alpha=0.9,
                       error_kw={"elinewidth": 1.5})
        bars2 = ax.bar(x + bar_width / 2, con_means, bar_width,
                       yerr=con_stds, capsize=4,
                       label="Constrained", color=BAR_DEF, alpha=0.9,
                       error_kw={"elinewidth": 1.5})

        for rect, val in zip(list(bars1) + list(bars2),
                             unc_means + con_means):
            if not np.isnan(val):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.015,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=FS_ANNOT - 1, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([DETECTOR_LABELS[d] for d in det_order],
                           fontsize=FS_TICK)
        ax.set_title(DATASET_LABELS[dataset], fontsize=FS_TITLE,
                     fontweight="bold")
        ax.set_xlabel("Anomaly Detector", fontsize=FS_LABEL)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND, frameon=True, edgecolor="black", fancybox=False)

    axes[0].set_ylabel("Attack Success Rate (ASR)", fontsize=FS_LABEL)
    fig.suptitle(
        "Physical Constraint Effect on PGD Attack Success",
        fontsize=FS_TITLE, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    out = figures_dir / "fig6_constraint_effect.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out).replace(".png", ".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 7: Within-Dataset Gradient Norm vs ASR (two-cluster scatter)
# ---------------------------------------------------------------------------

def plot_gradient_correlation(swat_grad_path: Path, smd_grad_path: Path,
                              figures_dir: Path):
    """
    Scatter plot showing gradient norm (attack_mean) vs known ASR for each
    detector × dataset combination, with separate regression lines per dataset.

    SWaT data comes from exp_a_gradient_within_swat.json.
    SMD data comes from smd/gradient_norms_smd.json.
    """
    # --- Load SWaT data ---
    swat_points = []  # list of (norm, asr, label, detector)
    if swat_grad_path.exists():
        swat_raw = _load_json(swat_grad_path)
        for det, info in swat_raw.items():
            norm = info.get("grad_norm_attack_mean",
                            info.get("grad_norm_all_mean", float("nan")))
            asr  = info.get("known_asr", float("nan"))
            swat_points.append((norm, asr, det, "swat"))
    else:
        print(f"  [WARN] fig7: {swat_grad_path.name} not found", file=sys.stderr)

    # --- Load SMD data ---
    smd_points = []
    if smd_grad_path.exists():
        smd_raw = _load_json(smd_grad_path)
        # ASR values (AutoAttack ε=0.05, from aggregate files)
        smd_asr = {"lstm_ae": 0.000, "tranad": 0.000, "usad": 0.045}
        for key, info in smd_raw.items():
            det = key.replace("_smd", "")
            norm = info.get("grad_norm", float("nan"))
            asr  = smd_asr.get(det, float("nan"))
            smd_points.append((norm, asr, det, "smd"))
    else:
        print(f"  [WARN] fig7: {smd_grad_path.name} not found", file=sys.stderr)

    if not swat_points and not smd_points:
        print("  [SKIP] fig7: no gradient data available")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    _apply_academic_style(ax)

    # Dataset visual encoding
    dataset_cfg = {
        "swat": {"marker": "o", "edgecolor": "navy",    "label": "SWaT"},
        "smd":  {"marker": "s", "edgecolor": "darkred", "label": "SMD"},
    }

    # --- Manual label offsets (dx, dy in offset points, ha) ---
    label_offsets = {
        ("tranad", "swat"):  (12, 12, "left"),
        ("lstm_ae", "swat"): (12, -18, "left"),
        ("usad", "swat"):    (12, 10, "left"),
        ("tranad", "smd"):   (0, -25, "center"),
        ("lstm_ae", "smd"):  (-40, 15, "right"),
        ("usad", "smd"):     (40, 15, "left"),
    }

    # --- Plot points ---
    for pts, dset_key in [(swat_points, "swat"), (smd_points, "smd")]:
        cfg = dataset_cfg[dset_key]
        for norm, asr, det, _ in pts:
            color = DETECTOR_COLORS.get(det, "#999999")
            ax.scatter(norm, asr, s=220, color=color, zorder=5,
                       marker=cfg["marker"],
                       edgecolors=cfg["edgecolor"], linewidths=1.2)
            det_label = DETECTOR_LABELS.get(det, det.upper())
            dx, dy, ha = label_offsets.get((det, dset_key), (0, 12, "center"))
            ax.annotate(
                det_label,
                (norm, asr),
                xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center",
                fontsize=FS_ANNOT + 1, fontweight="bold",
            )

    # --- Regression line per dataset ---
    reg_styles = {"swat": ("navy",    "SWaT fit"),
                  "smd":  ("darkred", "SMD fit")}
    for pts, dset_key in [(swat_points, "swat"), (smd_points, "smd")]:
        if len(pts) < 2:
            continue
        norms_arr = np.array([p[0] for p in pts])
        asrs_arr  = np.array([p[1] for p in pts])
        valid = ~(np.isnan(norms_arr) | np.isnan(asrs_arr))
        if valid.sum() < 2:
            continue
        slope, intercept, *_ = stats.linregress(norms_arr[valid], asrs_arr[valid])
        x_fit = np.linspace(norms_arr[valid].min() * 0.5,
                            norms_arr[valid].max() * 1.3, 200)
        y_fit = slope * x_fit + intercept
        color, reg_label = reg_styles[dset_key]
        ax.plot(x_fit, y_fit, "--", color=color, linewidth=1.8,
                alpha=0.7, label=reg_label)

    # --- Combined legend: detector colors + dataset shapes + regression lines ---
    det_handles = [
        mpatches.Patch(color=DETECTOR_COLORS[d], label=DETECTOR_LABELS[d])
        for d in SMD_DETECTOR_ORDER
    ]
    shape_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaa",
                   markeredgecolor="navy", markersize=9, label="SWaT points"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#aaa",
                   markeredgecolor="darkred", markersize=9, label="SMD points"),
        plt.Line2D([0], [0], linestyle="--", color="navy",
                   linewidth=1.5, label="SWaT fit (Spearman ρ=1.0)"),
        plt.Line2D([0], [0], linestyle="--", color="darkred",
                   linewidth=1.5, label="SMD fit"),
    ]
    leg1 = ax.legend(handles=det_handles, fontsize=FS_LEGEND,
                     loc="upper left", title="Detector",
                     title_fontsize=FS_LEGEND,
                     frameon=True, edgecolor="black", fancybox=False)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, fontsize=FS_LEGEND,
              loc="center right",
              frameon=True, edgecolor="black", fancybox=False)

    # --- Key insight annotation ---
    ax.text(0.35, 0.95,
            "SWaT: gradient-norm rank = ASR rank (ρ = 1.0)\n"
            "SMD: only USAD attains non-zero ASR\n"
            "Cross-dataset scales differ ~1000×",
            transform=ax.transAxes, fontsize=FS_ANNOT + 4,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F9F9F9",
                      ec="black", alpha=0.9, linewidth=0.7))

    ax.set_xlabel("Gradient Norm (attack samples)", fontsize=FS_LABEL)
    ax.set_ylabel("Attack Success Rate (ASR, AutoAttack ε=0.05)",
                  fontsize=FS_LABEL)
    ax.set_title(
        "Gradient Norm vs. ASR: Within-Dataset Ordering",
        fontsize=FS_TITLE, fontweight="bold"
    )
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(-0.05, 0.65)
    ax.set_xscale("log")
    fig.tight_layout()

    out = figures_dir / "fig7_gradient_correlation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out).replace(".png", ".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 8: Threshold Sensitivity (NEW)
# ---------------------------------------------------------------------------

def plot_threshold_sensitivity(threshold_data: dict, figures_dir: Path,
                               dataset: str = "swat"):
    if not threshold_data:
        print(f"  [SKIP] fig8: threshold_sweep_{dataset}.json not available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_academic_style(ax)

    BROKEN_DETS = {"dagmm", "gdn"}

    for det in DETECTOR_ORDER:
        if det not in threshold_data:
            continue
        entries = threshold_data[det]
        percentiles = [e["percentile"] for e in entries]
        f1_vals     = [e["f1"] for e in entries]
        color = DETECTOR_COLORS[det]
        is_broken = det in BROKEN_DETS
        # FIX: label must be the string DETECTOR_LABELS[det] only — no value prepended
        det_label = DETECTOR_LABELS[det]
        ax.plot(percentiles, f1_vals,
                marker="o",
                markersize=5 if is_broken else 7,
                linewidth=1.0 if is_broken else 2.0,
                linestyle="--" if is_broken else "-",
                alpha=0.6 if is_broken else 1.0,
                label=det_label,
                color=color)
        # Annotate only USAD's cliff (the key finding in this figure)
        if det == "usad" and len(f1_vals) >= 2:
            # Annotate the 99.5 point (before cliff) and 99.9 (after cliff)
            ax.annotate(
                f"F1={f1_vals[-2]:.2f}",
                (percentiles[-2], f1_vals[-2]),
                textcoords="offset points", xytext=(-60, 5),
                fontsize=FS_ANNOT, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
            )
            ax.annotate(
                f"F1={f1_vals[-1]:.2f}",
                (percentiles[-1], f1_vals[-1]),
                textcoords="offset points", xytext=(15, 20),
                fontsize=FS_ANNOT, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
            )

    ax.set_xlabel("Threshold Percentile", fontsize=FS_LABEL)
    ax.set_ylabel("F1 Score", fontsize=FS_LABEL)
    ax.set_title(
        f"Threshold Sensitivity: F1 vs. Percentile ({DATASET_LABELS[dataset]})",
        fontsize=FS_TITLE, fontweight="bold"
    )
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=FS_LEGEND, loc="lower left",
              frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()

    out = figures_dir / "fig8_threshold_sensitivity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out).replace(".png", ".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 9: Window Sensitivity (NEW)
# ---------------------------------------------------------------------------

def plot_window_sensitivity(results_dir: Path, figures_dir: Path):
    """
    Line/bar chart showing PGD ASR at ε=0.05 unconstrained across window
    sizes 50, 100, 200 for LSTM-AE, USAD, TranAD.
    Data read directly from the w50/w100/w200 sub-directories.
    """
    window_sizes = [50, 100, 200]
    det_order    = SMD_DETECTOR_ORDER  # same 3 detectors

    # Collect mean & std per detector × window
    data: dict[str, dict[str, float]] = {det: {} for det in det_order}

    for w in window_sizes:
        w_dir = results_dir / f"w{w}"
        for det in det_order:
            fname = f"attack_{det}_swat_pgd_0.05_unconstrained_aggregate.json"
            path  = w_dir / fname
            if path.exists():
                d = _load_json(path)
                asr = d.get("attack_success_rate", {})
                if isinstance(asr, dict):
                    data[det][w] = (asr.get("mean", float("nan")),
                                    asr.get("std",  0.0))
                else:
                    data[det][w] = (float(asr), 0.0)
            else:
                data[det][w] = (float("nan"), 0.0)

    fig, ax = plt.subplots(figsize=(9, 7))
    _apply_academic_style(ax)

    for det in det_order:
        means = [data[det].get(w, (float("nan"), 0.0))[0] for w in window_sizes]
        stds  = [data[det].get(w, (float("nan"), 0.0))[1] for w in window_sizes]
        color = DETECTOR_COLORS[det]
        label = DETECTOR_LABELS[det]
        ax.errorbar(
            window_sizes, means, yerr=stds,
            label=label, color=color,
            marker="o", markersize=9, linewidth=2.2,
            capsize=5, capthick=1.5
        )
        # Annotate each point — offset depends on detector to avoid overlap
        for w, m, s in zip(window_sizes, means, stds):
            if not np.isnan(m):
                # TranAD near zero: label above; LSTM-AE mid: label below; USAD high: label above error bar
                if det == "tranad":
                    dy = 14
                    va = "bottom"
                elif det == "usad":
                    dy = 16
                    va = "bottom"
                else:  # lstm_ae — label below to separate from USAD line
                    dy = -18
                    va = "top"
                ax.annotate(
                    f"{m:.3f}",
                    (w, m),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va=va, fontsize=FS_ANNOT,
                    color=color, fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1)
                )

    ax.set_xticks(window_sizes)
    ax.set_xticklabels([str(w) for w in window_sizes], fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_xlabel("Window Size", fontsize=FS_LABEL)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=FS_LABEL)
    ax.set_title(
        "Window Size Sensitivity: PGD ASR (ε = 0.05, SWaT, unconstrained)",
        fontsize=FS_TITLE, fontweight="bold"
    )
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=FS_LEGEND, loc="upper left",
              frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()

    out = figures_dir / "fig9_window_sensitivity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out).replace(".png", ".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------

def print_statistics(rows: list[dict], baselines: dict, grad_data: dict):
    """Print key statistics suitable for paper text."""
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("KEY STATISTICS FOR PAPER TEXT")
    print(SEP)

    # ---- Table 1: Baseline F1 scores ----
    print("\n[TABLE] Baseline F1 Scores (no attack)")
    header = f"{'Detector':<12}{'SWaT F1':>12}{'SMD F1':>12}"
    print(header)
    print("-" * 36)
    for det in DETECTOR_ORDER:
        swat_bl = baselines.get((det, "swat", ""), {})
        smd_bl  = baselines.get((det, "smd",  ""), {})
        sf = swat_bl.get("f1_mean", float("nan"))
        mf = smd_bl.get("f1_mean",  float("nan"))
        ss = swat_bl.get("f1_std",  0.0)
        ms = smd_bl.get("f1_std",   0.0)
        print(f"{DETECTOR_LABELS[det]:<12}"
              f"{sf:>9.4f}±{ss:.4f}"
              f"  {mf:>9.4f}±{ms:.4f}")

    # ---- Table 2: ASR heatmap at eps=0.05, unconstrained, SWaT ----
    print("\n\n[TABLE] ASR at ε=0.05 (unconstrained, SWaT)")
    header2 = f"{'Detector':<12}" + "".join(
        f"{ATTACK_LABELS[a]:>14}" for a in ATTACK_ORDER)
    print(header2)
    print("-" * (12 + 14 * len(ATTACK_ORDER)))
    for det in DETECTOR_ORDER:
        row_str = f"{DETECTOR_LABELS[det]:<12}"
        for atk in ATTACK_ORDER:
            matching = filter_rows(rows, detector=det, dataset="swat",
                                   attack=atk, epsilon="0.05",
                                   constraint="unconstrained")
            if matching:
                asr = matching[0]["asr_mean"]
                std = matching[0]["asr_std"]
                row_str += f"{asr:>10.3f}±{std:.3f}"
            else:
                row_str += f"{'N/A':>14}"
        print(row_str)

    # ---- Table 3: ASR heatmap at eps=0.05, unconstrained, SMD ----
    print("\n\n[TABLE] ASR at ε=0.05 (unconstrained, SMD)")
    print(header2)
    print("-" * (12 + 14 * len(ATTACK_ORDER)))
    for det in SMD_DETECTOR_ORDER:
        row_str = f"{DETECTOR_LABELS[det]:<12}"
        for atk in ATTACK_ORDER:
            matching = filter_rows(rows, detector=det, dataset="smd",
                                   attack=atk, epsilon="0.05",
                                   constraint="unconstrained")
            if matching:
                asr = matching[0]["asr_mean"]
                std = matching[0]["asr_std"]
                row_str += f"{asr:>10.3f}±{std:.3f}"
            else:
                row_str += f"{'N/A':>14}"
        print(row_str)

    # ---- Table 4: Defense effectiveness ----
    print("\n\n[TABLE] Defense Effectiveness (AT), ε=0.05, SWaT")
    print(f"{'Detector':<12}{'Undefended ASR':>18}{'Defended ASR':>16}{'ASR Drop':>12}")
    print("-" * 58)
    for det in DETECTOR_ORDER:
        attacks_avg = ["fgsm", "pgd"]
        u_vals = filter_rows(rows, detector=det, dataset="swat",
                             epsilon="0.05", constraint="unconstrained",
                             attack=attacks_avg)
        d_vals = filter_rows(rows, detector=det, dataset="swat",
                             constraint="defended", attack=attacks_avg)
        u_asrs = [r["asr_mean"] for r in u_vals if not np.isnan(r["asr_mean"])]
        d_asrs = [r["asr_mean"] for r in d_vals if not np.isnan(r["asr_mean"])]
        um = np.mean(u_asrs) if u_asrs else float("nan")
        dm = np.mean(d_asrs) if d_asrs else float("nan")
        drop = um - dm if not (np.isnan(um) or np.isnan(dm)) else float("nan")
        print(f"{DETECTOR_LABELS[det]:<12}"
              f"{um:>18.3f}"
              f"{dm:>16.3f}"
              f"{drop:>12.3f}")

    # ---- Table 5: Constraint effect on PGD ----
    print("\n\n[TABLE] Physical Constraint Effect on PGD ASR (ε=0.05, SWaT)")
    print(f"{'Detector':<12}{'Unconstrained':>16}{'Constrained':>14}{'ASR Drop':>12}")
    print("-" * 54)
    for det in DETECTOR_ORDER:
        unc = filter_rows(rows, detector=det, dataset="swat",
                          attack="pgd", epsilon="0.05",
                          constraint="unconstrained")
        con = filter_rows(rows, detector=det, dataset="swat",
                          attack="pgd", epsilon="0.05",
                          constraint="constrained")
        um = unc[0]["asr_mean"] if unc else float("nan")
        cm = con[0]["asr_mean"] if con else float("nan")
        drop = um - cm if not (np.isnan(um) or np.isnan(cm)) else float("nan")
        print(f"{DETECTOR_LABELS[det]:<12}"
              f"{um:>16.3f}"
              f"{cm:>14.3f}"
              f"{drop:>12.3f}")

    # ---- Gradient correlation ----
    if grad_data:
        print(f"\n\n[GRADIENT CORRELATION]")
        r_sq = grad_data.get("r_squared", float("nan"))
        pearson_r = grad_data.get("pearson_r", float("nan"))
        spearman_rho = grad_data.get("spearman_rho", float("nan"))
        print(f"  Pearson r    = {pearson_r:.4f}")
        print(f"  R²           = {r_sq:.4f}")
        print(f"  Spearman ρ   = {spearman_rho:.4f}")
        print(f"\n  Gradient norms per detector-dataset:")
        for k, v in sorted(grad_data.get("gradient_norms", {}).items()):
            asr_v = grad_data.get("asr", {}).get(k, float("nan"))
            print(f"    {k:<20s}  norm={v:.6f}   ASR={asr_v:.3f}")

    # ---- Summary sentences for paper ----
    print(f"\n\n[INLINE STATISTICS FOR PAPER]")
    # Most vulnerable: highest mean ASR across all eps=0.05, unconstrained, SWaT
    swat_unc = filter_rows(rows, dataset="swat", constraint="unconstrained",
                           epsilon="0.05")
    if swat_unc:
        # group by detector
        by_det = {}
        for r in swat_unc:
            by_det.setdefault(r["detector"], []).append(r["asr_mean"])
        det_mean_asr = {d: np.nanmean(v) for d, v in by_det.items()}
        most_vuln = max(det_mean_asr, key=det_mean_asr.get)
        most_robust = min(det_mean_asr, key=det_mean_asr.get)
        print(f"  Most vulnerable detector (SWaT, ε=0.05): "
              f"{DETECTOR_LABELS[most_vuln]} (mean ASR={det_mean_asr[most_vuln]:.3f})")
        print(f"  Most robust detector (SWaT, ε=0.05):     "
              f"{DETECTOR_LABELS[most_robust]} (mean ASR={det_mean_asr[most_robust]:.3f})")

    # Best attack overall
    by_atk = {}
    for r in filter_rows(rows, constraint="unconstrained", epsilon="0.05"):
        by_atk.setdefault(r["attack"], []).append(r["asr_mean"])
    if by_atk:
        atk_means = {a: np.nanmean(v) for a, v in by_atk.items()}
        best_atk = max(atk_means, key=atk_means.get)
        print(f"  Strongest attack overall (ε=0.05, unconstrained): "
              f"{ATTACK_LABELS[best_atk]} (mean ASR={atk_means[best_atk]:.3f})")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures for ICS adversarial robustness paper"
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Directory containing *_aggregate.json files")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR,
                        help="Output directory for figures")
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    figures_dir: Path = args.figures_dir

    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    figures_dir.mkdir(parents=True, exist_ok=True)

    smd_dir = results_dir / "smd"

    # ---- Load data ----
    print("Loading data...")
    rows      = load_attack_results(results_dir)
    smd_rows  = load_attack_results(smd_dir) if smd_dir.exists() else []
    all_rows  = rows + smd_rows
    baselines = load_baseline_results(results_dir)
    smd_bl    = load_baseline_results(smd_dir) if smd_dir.exists() else {}
    baselines.update(smd_bl)
    grad_data = load_gradient_norms(results_dir)
    ts_swat   = load_threshold_sweep(results_dir, "swat")

    swat_grad_path = results_dir / "exp_a_gradient_within_swat.json"
    smd_grad_path  = smd_dir / "gradient_norms_smd.json"

    # ---- Figures ----
    print("\nGenerating figures...")

    print("  [fig2] ASR heatmap — SWaT")
    plot_asr_heatmap(all_rows, "swat", figures_dir)

    print("  [fig3] ASR heatmap — SMD")
    plot_asr_heatmap(all_rows, "smd", figures_dir)

    print("  [fig4] DR vs epsilon — SWaT / PGD")
    plot_epsilon_dr(all_rows, figures_dir)

    print("  [fig5] Defense comparison (SWaT + SMD)")
    plot_defense_comparison(all_rows, figures_dir)

    print("  [fig6] Constraint effect (SWaT + SMD)")
    plot_constraint_effect(all_rows, figures_dir)

    print("  [fig7] Gradient norm vs ASR scatter (within-dataset)")
    plot_gradient_correlation(swat_grad_path, smd_grad_path, figures_dir)

    print("  [fig8] Threshold sensitivity")
    plot_threshold_sensitivity(ts_swat, figures_dir, dataset="swat")

    print("  [fig9] Window sensitivity")
    plot_window_sensitivity(results_dir, figures_dir)

    # ---- Statistics ----
    print_statistics(all_rows, baselines, grad_data)

    # ---- Verify outputs ----
    print("Verifying outputs...")
    expected = [
        "fig2_heatmap_swat.png",
        "fig3_heatmap_smd.png",
        "fig4_epsilon_dr_swat.png",
        "fig5_defense_comparison.png",
        "fig6_constraint_effect.png",
        "fig7_gradient_correlation.png",
        "fig8_threshold_sensitivity.png",
        "fig9_window_sensitivity.png",
    ]
    all_ok = True
    for fname in expected:
        p = figures_dir / fname
        if p.exists() and p.stat().st_size > 10_000:
            print(f"  OK   {fname}  ({p.stat().st_size // 1024} KB)")
        else:
            status = "MISSING" if not p.exists() else "TOO SMALL"
            print(f"  FAIL {fname}  [{status}]")
            all_ok = False

    if all_ok:
        print("\nAll figures generated successfully.")
    else:
        print("\nWARNING: Some figures were not generated correctly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
