#!/usr/bin/env python3
"""Fig 1: Benchmark framework — academic style, all Unicode (no LaTeX)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 0,
    "svg.fonttype": "none",
})

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-1, 8.5)
ax.axis("off")

# Colors
C_BORDER = "#333333"
C_ATTACK = "#B91C1C"
C_DEFENSE= "#1D4ED8"
C_GRAD   = "#047857"
C_TEXT   = "#1F2937"

LW = 1.5
FS_TITLE = 13
FS_BODY  = 11
FS_LABEL = 10


def box(x, y, w, h, label, sublabels=None, border=C_BORDER, fill="white", bold=True):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                          facecolor=fill, edgecolor=border, linewidth=LW)
    ax.add_patch(rect)
    ty = y + h - 0.3 if sublabels else y + h / 2
    ax.text(x + w / 2, ty, label, ha="center", va="center",
            fontsize=FS_TITLE, fontweight="bold" if bold else "normal",
            color=C_TEXT)
    if sublabels:
        for i, s in enumerate(sublabels):
            ax.text(x + w / 2, ty - 0.4 - i * 0.35, s,
                    ha="center", va="center", fontsize=FS_BODY, color="#6B7280")


def arrow(x1, y1, x2, y2, label=None, color=C_BORDER, lw=1.5, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, shrinkA=2, shrinkB=2))
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        offset = (0, 8) if x1 != x2 else (-12, 0)
        ax.annotate(label, (mx, my), textcoords="offset points",
                    xytext=offset, ha="center", va="center",
                    fontsize=FS_LABEL, color=color, style="italic")


# ── MAIN FLOW ──

box(0, 3.2, 2.2, 1.6, "ICS Plant", ["SWaT / SMD"])

box(3, 3.2, 2.2, 1.6, "Sensors",
    ["x(t) ∈ ℝᵈ"])

box(6, 2.7, 3.0, 2.6, "Anomaly Detector",
    ["s(x) = ‖x − f_θ(x)‖²",
     "alert if s(x) > τ",
     "",
     "LSTM-AE | USAD | TranAD"])

box(10, 3.2, 2.2, 1.6, "Evaluation", ["TP / FP / DR / ASR"])

arrow(2.2, 4.0, 3.0, 4.0, "x(t)")
arrow(5.2, 4.0, 6.0, 4.0, "x(t)")
arrow(9.0, 4.0, 10.0, 4.0, "ŷ vs y")

# ── ADVERSARIAL ATTACKER ──

box(3.5, 6.5, 5.0, 1.5, "Adversarial Attacker",
    ["x̃ = x + δ,   ‖δ‖∞ ≤ ε"],
    border=C_ATTACK, fill="#FEF2F2")

arrow(6.0, 6.5, 6.8, 5.3, "x̃(t)", color=C_ATTACK, lw=2.0)

# ── PHYSICAL CONSTRAINTS ──

box(0, 6.5, 3.0, 1.5, "Physical Constraints",
    ["lᵢ ≤ x̃ᵢ ≤ uᵢ",
     "|δₜ − δₜ₋₁| ≤ Δmax"],
    border="#6B7280", fill="#F9FAFB")

arrow(3.0, 7.25, 3.5, 7.25, color="#6B7280")

# ── ADVERSARIAL TRAINING ──

box(5.0, 0, 4.0, 1.5, "Adversarial Training (AT)",
    ["min_θ E[ L(x+δ, f_θ) ]"],
    border=C_DEFENSE, fill="#EFF6FF")

arrow(7.0, 1.5, 7.0, 2.7, "hardens", color=C_DEFENSE, lw=2.0)

# ── GRADIENT NORM ANALYSIS ──

box(9.5, 6.5, 3.0, 1.5, "Gradient Norm Analysis",
    ["ḡ = mean ‖∇ₓ s(x)‖₂"],
    border=C_GRAD, fill="#ECFDF5")

arrow(8.5, 5.3, 10.0, 6.5, color=C_GRAD, lw=1.5)

ax.text(11.0, 6.25, "low ḡ → robust",
        fontsize=FS_LABEL, color=C_GRAD, ha="center")
ax.text(11.0, 5.95, "high ḡ → vulnerable",
        fontsize=FS_LABEL, color=C_GRAD, ha="center")

# ── DASHED SCOPE ──

scope = mpatches.FancyBboxPatch(
    (2.7, 2.4), 7.0, 6.0,
    boxstyle="round,pad=0.3",
    facecolor="none", edgecolor="#9CA3AF",
    linewidth=1.0, linestyle="--")
ax.add_patch(scope)
ax.text(6.2, 8.55, "Benchmark Evaluation Scope",
        ha="center", fontsize=FS_LABEL, color="#9CA3AF", style="italic")

# Journal figure style: border line around the figure.
frame = mpatches.Rectangle(
    (-0.5, -1), 16.0, 9.9,
    facecolor="none", edgecolor="#333333", linewidth=0.8,
    clip_on=False)
ax.add_patch(frame)

fig.tight_layout(pad=0.5)
fig.savefig("fig1_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("fig1_architecture.svg", format="svg", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig1_architecture.svg + .png")
