"""
Read the CSV files produced by codes/classification_vgg16.py and plot
baseline vs. imputed downstream classification performance.

Expected input files under ./results/classification/:
    {dataset}_baseline_vgg16_classification.csv
    {dataset}_{model_impt}_{mechanism}_vgg16_classification.csv

Each file has one row per fold (fold0..fold4) and one column per metric
(ACCURACY, F1, PRECISION, RECALL, AUC).

Produces, under ./results/classification/plots/:
    - one grouped bar chart per (dataset, mechanism): x = metric,
      bars = baseline + each imputer, height = mean across folds,
      error bars = std across folds.
    - overall_vgg16_classification.png: the same grouped bar chart, but
      pooled across every available dataset and mechanism at once, for a
      single at-a-glance ranking of methods.
    - classification_results_long.csv / classification_results_summary.csv:
      tidy long-format and mean+-std aggregated tables for further analysis.
"""

import sys

sys.path.append("./")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_MECHANISMS = {
    #"inbreast": ["MNAR-SQUARES", "MNAR-LINES"],
    #"mias": ["MNAR-SQUARES", "MNAR-LINES"],
    "vindr-reduzido": ["MNAR-SQUARES", "MNAR-LINES"],
    #"cbis-ddsm": ["MCAR"],
}
IMPUTERS = ["knn", "mc", "vaewl", "mae-vit", "dip", "diffusion", "mat", "harp"]
METRICS = ["ACCURACY", "F1", "AUC"]

# Fixed categorical order (slots 1, 2, 3, 4, 5, 6 of the validated default
# palette, references/palette.md) -- never reassigned/cycled per chart, so a
# given method always keeps the same color across every figure. Baseline
# uses the muted-ink gray instead of a categorical hue: it is the reference
# condition, not one of the methods being compared (emphasis pattern).
METHOD_COLORS = {
    "baseline": "#898781",
    "knn": "#2a78d6",
    "mc": "#1baf7a",
    "vaewl": "#eda100",
    "mae-vit": "#008300",
    "dip": "#4a3aa7",
    "diffusion": "#e34948",
    "mat": "#eb6834",
    "harp": "#e87ba4",
}
METHOD_LABELS = {
    "baseline": "Baseline (clean)",
    "knn": "kNN",
    "mc": "MC",
    "vaewl": "VAE-WL",
    "mae-vit": "CMask-ViT",
    "dip": "DIP",
    "diffusion": "Diffusion",
    "mat": "MAT",
    "harp": "HARP",
}
METRIC_LABELS = {
    "ACCURACY": "Accuracy",
    "F1": "F1",
    "AUC": "AUC",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def load_long_results(results_dir: str = "./results/classification") -> pd.DataFrame:
    """Collect every available fold/metric value into a tidy long-format table."""
    rows = []

    for dataset, mechanisms in DATASET_MECHANISMS.items():
        baseline_path = os.path.join(
            results_dir, f"{dataset}_baseline_vgg16_classification.csv"
        )
        if not os.path.exists(baseline_path):
            continue
        baseline_df = pd.read_csv(baseline_path, index_col=0)

        for mechanism in mechanisms:
            # Baseline is mechanism-agnostic (clean test images), replicated
            # here so it can be plotted side by side with each mechanism's
            # imputers for a fair, like-for-like comparison.
            for fold, row in baseline_df.iterrows():
                for metric in METRICS:
                    rows.append(
                        {
                            "DATASET": dataset,
                            "MECHANISM": mechanism,
                            "METHOD": "baseline",
                            "FOLD": fold,
                            "METRIC": metric,
                            "VALUE": row[metric],
                        }
                    )

            for model_impt in IMPUTERS:
                path = os.path.join(
                    results_dir,
                    f"{dataset}_{model_impt}_{mechanism}_vgg16_classification.csv",
                )
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path, index_col=0)
                for fold, row in df.iterrows():
                    for metric in METRICS:
                        rows.append(
                            {
                                "DATASET": dataset,
                                "MECHANISM": mechanism,
                                "METHOD": model_impt,
                                "FOLD": fold,
                                "METRIC": metric,
                                "VALUE": row[metric],
                            }
                        )

    return pd.DataFrame(rows)


def _plot_grouped_bar(subset: pd.DataFrame, title: str, output_path: str) -> bool:
    """
    Shared grouped-bar renderer: x = metric, bars = baseline + whichever
    imputers are present in `subset`. Returns False (and draws nothing) if
    there's nothing to compare against baseline.
    """
    methods = [m for m in ["baseline"] + IMPUTERS if m in subset["METHOD"].unique()]
    n_methods = len(methods)
    if n_methods <= 1:
        return False

    summary = subset.groupby(["METHOD", "METRIC"])["VALUE"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bar_width = 0.8 / n_methods
    x = np.arange(len(METRICS))

    for i, method in enumerate(methods):
        means = [summary.loc[(method, metric), "mean"] if (method, metric) in summary.index else np.nan for metric in METRICS]
        stds = [summary.loc[(method, metric), "std"] if (method, metric) in summary.index else 0.0 for metric in METRICS]
        offset = (i - (n_methods - 1) / 2) * bar_width

        ax.bar(
            x + offset,
            means,
            width=bar_width * 0.9,
            yerr=stds,
            capsize=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            zorder=3,
        )

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], color=INK_SECONDARY)
    ax.tick_params(axis="y", colors=INK_SECONDARY)
    ax.set_ylabel("Score", color=INK_SECONDARY)

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine_name in ["top", "right", "left"]:
        ax.spines[spine_name].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_AXIS)

    #ax.set_title(title, color=INK_PRIMARY, fontsize=13, loc="left")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(n_methods, 4),
        frameon=False,
        labelcolor=INK_SECONDARY,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_dataset_mechanism(
    df_long: pd.DataFrame, dataset: str, mechanism: str, output_dir: str
):
    """Grouped bar chart: x = metric, bars = baseline + each imputer, for one (dataset, mechanism)."""
    subset = df_long[(df_long["DATASET"] == dataset) & (df_long["MECHANISM"] == mechanism)]
    if subset.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    drawn = _plot_grouped_bar(
        subset,
        f"VGG16 downstream classification -- {dataset} / {mechanism}",
        os.path.join(output_dir, f"{dataset}_{mechanism}_vgg16_classification.png"),
    )
    if not drawn:
        print(f"Skipping {dataset}/{mechanism}: no imputer results found yet (baseline only).")


def plot_overall(df_long: pd.DataFrame, output_dir: str):
    """Single grouped bar chart pooled across every available dataset and mechanism."""
    if df_long.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    drawn = _plot_grouped_bar(
        df_long,
        "VGG16 downstream classification -- overall (all datasets & mechanisms)",
        os.path.join(output_dir, "overall_vgg16_classification.png"),
    )
    if not drawn:
        print("Skipping overall plot: no imputer results found yet (baseline only).")


def main(
    results_dir: str = "./results/classification",
    output_dir: str = "./results/classification/plots",
):
    df_long = load_long_results(results_dir)
    if df_long.empty:
        print(f"No classification results found under '{results_dir}'.")
        return

    os.makedirs(results_dir, exist_ok=True)
    df_long.to_csv(os.path.join(results_dir, "classification_results_long.csv"), index=False)

    summary = (
        df_long.groupby(["DATASET", "MECHANISM", "METHOD", "METRIC"])["VALUE"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(os.path.join(results_dir, "classification_results_summary.csv"), index=False)

    for dataset, mechanisms in DATASET_MECHANISMS.items():
        for mechanism in mechanisms:
            plot_dataset_mechanism(df_long, dataset, mechanism, output_dir)

    plot_overall(df_long, output_dir)

    print(f"Plots written to '{output_dir}'.")


if __name__ == "__main__":
    main()
