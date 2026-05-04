#!/usr/bin/env python3
"""
Generate all publication figures for the compliance estimation paper.

Usage:
    python scripts/make_paper_figures.py [--data-dir DIR] [--out-dir DIR]

Defaults:
    --data-dir  tests/branch_compliance_analysis_fg
    --out-dir   figures/
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── colour palette ────────────────────────────────────────────────────────────
C_PROX = "#2166ac"
C_MID  = "#4dac26"
C_DIST = "#d7191c"
ZONE_COLORS  = {"proximal": C_PROX, "mid": C_MID, "distal": C_DIST}
ZONE_LABELS  = {"proximal": "Proximal", "mid": "Mid", "distal": "Distal"}
C_RAW        = "#e66101"
C_SMOOTHED   = "#4393c3"
C_EXCITATION = "#762a83"

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def zone_color(row):
    return ZONE_COLORS.get(row["distance_zone"], "gray")


def load_data(data_dir: Path):
    def _csv(name):
        return pd.read_csv(data_dir / name)

    comp  = _csv("compliance_summary.csv")
    bend  = _csv("bending_metrics.csv")
    node  = _csv("node_metrics.csv")
    edge  = _csv("edge_metrics.csv")
    zone  = _csv("zone_summary.csv")
    smb   = _csv("smoothing_bending_comparison.csv")
    sme   = _csv("smoothing_edge_comparison.csv")
    excit = _csv("relative_compliance_excitation_timeseries.csv")
    return comp, bend, node, edge, zone, smb, sme, excit


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 in paper: Smoothing effect (raw vs FG-smoothed)
# ─────────────────────────────────────────────────────────────────────────────
def fig_smoothing(smb, sme, out_path: Path):
    """
    Two-panel figure showing the effect of factor-graph smoothing.

    Left (a): Bending angle RMS vs root distance — raw vs smoothed.
    Right (b): Edge length CV vs root distance — raw vs smoothed.

    Corresponds to Fig. 4 in the paper.
    """
    smb_raw  = smb.dropna(subset=["filtered_angle_rms_delta_deg_raw"]).sort_values("root_distance_px")
    smb_sm   = smb.dropna(subset=["filtered_angle_rms_delta_deg_smoothed"]).sort_values("root_distance_px")
    sme_raw  = sme.dropna(subset=["length_cv_raw"]).sort_values("root_distance_px")
    sme_sm   = sme.dropna(subset=["length_cv_smoothed"]).sort_values("root_distance_px")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    ax = axes[0]
    ax.plot(smb_raw["root_distance_px"], smb_raw["filtered_angle_rms_delta_deg_raw"],
            "o--", color=C_RAW, ms=4, lw=1.2, label="Raw tracks")
    ax.plot(smb_sm["root_distance_px"], smb_sm["filtered_angle_rms_delta_deg_smoothed"],
            "s-", color=C_SMOOTHED, ms=4, lw=1.2, label="Smoothed")
    ax.set_xlabel("Root distance (px)")
    ax.set_ylabel(r"Bending angle RMS (deg)")
    ax.set_title("(a) Bending: raw vs. smoothed tracks", fontweight="bold")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(sme_raw["root_distance_px"], sme_raw["length_cv_raw"],
             "o--", color=C_RAW, ms=4, lw=1.2, label="Raw tracks")
    ax2.plot(sme_sm["root_distance_px"], sme_sm["length_cv_smoothed"],
             "s-", color=C_SMOOTHED, ms=4, lw=1.2, label="Smoothed")
    ax2.set_xlabel("Root distance (px)")
    ax2.set_ylabel("Edge length CV (dimensionless)")
    ax2.set_title("(b) Inextensibility: raw vs. smoothed", fontweight="bold")
    ax2.legend()

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 in paper: Motion accumulation + per-joint bending with CI
# ─────────────────────────────────────────────────────────────────────────────
def fig_motion(node, bend, out_path: Path):
    """
    Two-panel figure.

    Left (a): Node relative motion RMS vs root distance, zone-colored.
    Right (b): Per-joint bending angle RMS Δβ with 95% CI, zone-colored.

    Corresponds to Fig. 5 in the paper.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # ── (a) motion accumulation ───────────────────────────────────────────────
    ax = axes[0]
    for _, r in node.iterrows():
        c = zone_color(r)
        ax.scatter(r["root_distance_px"], r["rel_motion_rms_px"],
                   color=c, s=35, edgecolors="white", linewidths=0.4, zorder=5)
        lo, hi = r.get("rel_motion_rms_ci_low", np.nan), r.get("rel_motion_rms_ci_high", np.nan)
        if pd.notna(lo) and pd.notna(hi):
            ax.plot([r["root_distance_px"]] * 2, [lo, hi], color=c, lw=1.0, alpha=0.4)
    ax.set_xlabel("Root distance (px)")
    ax.set_ylabel("Relative motion RMS (px)")
    ax.set_title("(a) Root-to-tip motion accumulation", fontweight="bold")
    patches = [mpatches.Patch(color=ZONE_COLORS[z], label=ZONE_LABELS[z])
               for z in ["proximal", "mid", "distal"]]
    ax.legend(handles=patches, loc="upper left")

    # ── (b) per-joint bending with CI ────────────────────────────────────────
    ax2 = axes[1]
    bend_s = bend.sort_values("root_distance_px")
    for _, r in bend_s.iterrows():
        c = zone_color(r)
        ax2.scatter(r["root_distance_px"], r["filtered_angle_rms_delta_deg"],
                    color=c, s=35, edgecolors="white", linewidths=0.4, zorder=5)
        lo, hi = r.get("filtered_angle_rms_ci_low", np.nan), r.get("filtered_angle_rms_ci_high", np.nan)
        if pd.notna(lo) and pd.notna(hi):
            ax2.plot([r["root_distance_px"]] * 2, [lo, hi], color=c, lw=1.0, alpha=0.4)
    ax2.set_xlabel("Root distance (px)")
    ax2.set_ylabel(r"Bending angle RMS $\Delta\beta$ (deg)")
    ax2.set_title("(b) Per-joint bending deviation with CI", fontweight="bold")

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 in paper: Compliance map + bootstrap uncertainty
# ─────────────────────────────────────────────────────────────────────────────
def fig_compliance(comp, out_path: Path):
    """
    Two-panel figure.

    Left (a): κe per edge vs root distance with 95% CI, zone-colored.
    Right (b): Bootstrap relative CI width per edge (bar chart), zone-colored.

    Corresponds to Fig. 6 in the paper.
    """
    # exclude root edge (NaN kappa) – depth=1 has no bending triplet
    cp = comp.dropna(subset=["relative_compliance_kappa"]).copy()
    cp = cp[cp["depth"] > 1].sort_values("root_distance_px").reset_index(drop=True)

    x   = cp["root_distance_px"].values
    k   = cp["relative_compliance_kappa"].values
    klo = cp["relative_compliance_ci_low"].values
    khi = cp["relative_compliance_ci_high"].values
    colors = [zone_color(r) for _, r in cp.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # ── (a) compliance scatter with CI ───────────────────────────────────────
    ax = axes[0]
    for xi, ki, lo, hi, c in zip(x, k, klo, khi, colors):
        ax.plot([xi, xi], [lo, hi], color=c, lw=1.2, alpha=0.5)
    ax.scatter(x, k, c=colors, s=40, zorder=5, edgecolors="white", linewidths=0.4)
    ax.axhline(1.0, ls="--", lw=0.8, color="k", alpha=0.5)
    ax.set_xlabel("Root distance (px)")
    ax.set_ylabel(r"Relative compliance $\kappa_e$")
    ax.set_title("(a) Per-edge compliance with 95% CI", fontweight="bold")
    ax.set_ylim(0, 3.2)
    patches = [mpatches.Patch(color=ZONE_COLORS[z], label=ZONE_LABELS[z])
               for z in ["proximal", "mid", "distal"]]
    ax.legend(handles=patches, loc="upper left")

    # ── (b) bootstrap uncertainty bars ───────────────────────────────────────
    ax2 = axes[1]
    u = (khi - klo) / k
    bars = ax2.bar(range(len(cp)), u, color=colors, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Edge index (root → tip)")
    ax2.set_ylabel(r"Relative CI width $(CI_{hi} - CI_{lo})/\kappa_e$")
    ax2.set_title("(b) Bootstrap uncertainty per edge", fontweight="bold")
    ax2.set_xticks(range(len(cp)))
    # label each bar with edge depth (= e2, e3, ..., e19 for a linear chain)
    ax2.set_xticklabels([f"e{int(r['depth'])}" for _, r in cp.iterrows()],
                        rotation=45, fontsize=5)

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 in paper: Shared excitation proxy over time
# ─────────────────────────────────────────────────────────────────────────────
def fig_excitation(excit, out_path: Path):
    """
    Single-panel time series of estimated shared excitation proxy z_t.

    Corresponds to Fig. 7 in the paper.
    """
    # column is always 'excitation_proxy_deg' (from analyze_branch_compliance.py)
    zcol = "excitation_proxy_deg"
    if zcol not in excit.columns:
        # fallback: second column
        zcol = excit.columns[1]

    fig, ax = plt.subplots(figsize=(7.16, 2.4))
    ax.plot(excit["frame"], excit[zcol], color=C_EXCITATION, lw=1.0, alpha=0.9)
    ax.set_xlabel("Frame")
    ax.set_ylabel(r"Shared excitation proxy $z_t$ (deg)")
    ax.set_title("Estimated global airflow excitation over time", fontweight="bold")

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary: Zone bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_zones(zone, out_path: Path):
    """
    Three-panel bar chart aggregated by anatomical zone.
    Not a paper figure; included as a supplementary summary.
    """
    zone_d = {r["distance_zone"]: r for _, r in zone.iterrows()}
    zones = [z for z in ["proximal", "mid", "distal"] if z in zone_d]

    metrics = [
        ("relative_compliance_kappa_mean", r"Mean $\kappa_e$"),
        ("node_rel_motion_rms_mean",         "Mean node motion RMS (px)"),
        ("filtered_angle_rms_mean",          r"Mean bending RMS $\Delta\beta$ (°)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    for ax, (col, label) in zip(axes, metrics):
        vals   = [zone_d[z][col] for z in zones]
        cols   = [ZONE_COLORS[z] for z in zones]
        znames = [ZONE_LABELS[z] for z in zones]
        bars   = ax.bar(znames, vals, color=cols, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(vals),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    axes[0].axhline(1.0, ls="--", lw=0.8, color="k", alpha=0.5)
    fig.suptitle("Zone-averaged metrics", fontweight="bold")
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="tests/branch_compliance_analysis_fg",
                        help="Directory containing the analysis output CSVs")
    parser.add_argument("--out-dir",  default="figures",
                        help="Output directory for generated figures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    comp, bend, node, edge, zone, smb, sme, excit = load_data(data_dir)

    print("Generating figures...")
    fig_smoothing(smb,  sme,  out_dir / "fig4_smoothing.pdf")
    fig_motion   (node, bend, out_dir / "fig5_motion.pdf")
    fig_compliance(comp,      out_dir / "fig6_compliance.pdf")
    fig_excitation(excit,     out_dir / "fig7_excitation.pdf")
    fig_zones     (zone,      out_dir / "figS_zones.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
