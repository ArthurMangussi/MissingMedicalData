"""
Read the CMask-ViT masking-strategy ablation results produced by
codes/ablation_mae_vit_masking.py (e.g. results.csv, columns: DATASET,
MECHANISM, FOLD, MASKING_STRATEGY, MASKED_PATCH_FRACTION_MEAN, MAE, PSNR,
SSIM) and visualize how each masking strategy performs across the three
missingness mechanisms.

Produces, under ./results/ablation_mae_vit/plots/:
    - ablation_summary_table.csv: mean +- std over folds, per
      (mechanism, masking strategy), for every metric.
    - ablation_metrics_grid.png: PSNR / SSIM / MAE x mechanism grid,
      one bar chart per cell, one bar per masking strategy.
    - ablation_masked_fraction.png: mean fraction of patches masked per
      strategy and mechanism -- explains WHY the metrics differ (e.g.
      any_pixel masks ~99% of patches under MCAR, but ~2% under
      MNAR-SQUARES, because the aggregation rule interacts very
      differently with scattered vs. block-shaped corruption).
"""

import sys

sys.path.append("./")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGY_ORDER = [
    "any_pixel_proposed",
    "threshold_25",
    "threshold_50",
    "vanilla_random_ratio_075",
]
STRATEGY_LABELS = {
    "any_pixel_proposed": "Ours",
    "threshold_25": "Threshold\n≥25%",
    "threshold_50": "Threshold\n≥50%",
    "vanilla_random_ratio_075": "Vanilla MAE\n(random, r=0.75)",
}
# Fixed categorical order (validated default palette) -- the proposed
# method keeps the first (most prominent) slot; a given strategy always
# keeps the same color across every subplot and every figure.
STRATEGY_COLORS = {
    "any_pixel_proposed": "#2a78d6",
    "threshold_25": "#1baf7a",
    "threshold_50": "#eda100",
    "threshold_75": "#008300",
    "random_matched": "#4a3aa7",
    "vanilla_random_ratio_075": "#e87ba4",
}
MECHANISM_ORDER = ["MCAR", "MNAR-SQUARES", "MNAR-LINES"]
MECHANISM_LABELS = {
    "MCAR": "MCAR (dead pixels)",
    "MNAR-SQUARES": "MAR (squares)",
    "MNAR-LINES": "MNAR (stripes)",
}
MECHANISM_COLORS = {
    "MCAR": "#2a78d6",
    "MNAR-SQUARES": "#eda100",
    "MNAR-LINES": "#e34948",
}
METRIC_ROWS = [
    ("PSNR", "PSNR, dB (higher better)"),
    ("SSIM", "SSIM (higher better)"),
    ("MAE", "MAE (lower better)"),
]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def load_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    summary = df.groupby(["MECHANISM", "MASKING_STRATEGY"])[
        ["MAE", "PSNR", "SSIM", "MASKED_PATCH_FRACTION_MEAN"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def plot_metrics_grid(summary: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(
        len(METRIC_ROWS), len(MECHANISM_ORDER), figsize=(15, 11), facecolor=SURFACE
    )

    for row_i, (metric, metric_label) in enumerate(METRIC_ROWS):
        for col_i, mechanism in enumerate(MECHANISM_ORDER):
            ax = axes[row_i, col_i]
            ax.set_facecolor(SURFACE)

            subset = summary[summary["MECHANISM"] == mechanism].set_index("MASKING_STRATEGY")

            x = np.arange(len(STRATEGY_ORDER))
            means = [
                subset.loc[s, f"{metric}_mean"] if s in subset.index else np.nan
                for s in STRATEGY_ORDER
            ]
            stds = [
                subset.loc[s, f"{metric}_std"] if s in subset.index else 0.0
                for s in STRATEGY_ORDER
            ]
            colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]

            ax.bar(x, means, yerr=stds, capsize=2, color=colors, width=0.7, zorder=3)

            ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
            ax.set_axisbelow(True)
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(BASELINE_AXIS)
            ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=8)

            if row_i == 0:
                ax.set_title(MECHANISM_LABELS[mechanism], color=INK_PRIMARY, fontsize=12)
            if col_i == 0:
                ax.set_ylabel(metric_label, color=INK_SECONDARY, fontsize=9)

            ax.set_xticks(x)
            if row_i == len(METRIC_ROWS) - 1:
                ax.set_xticklabels(
                    [STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
                    rotation=45,
                    ha="right",
                    fontsize=7.5,
                    color=INK_SECONDARY,
                )
            else:
                ax.set_xticklabels([])

    # fig.suptitle(
    #     "MAE-ViT masking-strategy ablation -- mean ± std over folds",
    #     color=INK_PRIMARY,
    #     fontsize=14,
    #     x=0.02,
    #     ha="left",
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_masked_fraction(summary: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    n_mechanisms = len(MECHANISM_ORDER)
    bar_width = 0.8 / n_mechanisms
    x = np.arange(len(STRATEGY_ORDER))

    for i, mechanism in enumerate(MECHANISM_ORDER):
        subset = summary[summary["MECHANISM"] == mechanism].set_index("MASKING_STRATEGY")
        means = [
            subset.loc[s, "MASKED_PATCH_FRACTION_MEAN_mean"] if s in subset.index else np.nan
            for s in STRATEGY_ORDER
        ]
        offset = (i - (n_mechanisms - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means,
            width=bar_width * 0.9,
            color=MECHANISM_COLORS[mechanism],
            label=MECHANISM_LABELS[mechanism],
            zorder=3,
        )

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
        rotation=45,
        ha="right",
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.set_ylabel("Mean fraction of patches masked", color=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.set_title(
        "Masking budget by strategy and mechanism", color=INK_PRIMARY, fontsize=13, loc="left"
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=3,
        frameon=False,
        labelcolor=INK_SECONDARY,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main(csv_path: str = "./results.csv", output_dir: str = "./results/ablation_mae_vit/plots"):
    os.makedirs(output_dir, exist_ok=True)
    summary = load_summary(csv_path)

    summary.to_csv(os.path.join(output_dir, "ablation_summary_table.csv"), index=False)
    plot_metrics_grid(summary, os.path.join(output_dir, "ablation_metrics_grid.png"))
    plot_masked_fraction(summary, os.path.join(output_dir, "ablation_masked_fraction.png"))

    print(f"Table and plots written to '{output_dir}'.")


if __name__ == "__main__":
    main()
