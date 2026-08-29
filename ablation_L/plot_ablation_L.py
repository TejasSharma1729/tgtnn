#!/usr/bin/env python3
"""
Two ablation plots from ablation_L_results.csv.

Plot 1: L  ablation — mean query time vs L  (Lq=5 fixed),  2×2 grid per dataset.
Plot 2: Lq ablation — mean query time vs Lq (L=10 fixed),  2×2 grid per dataset.

Saves ablation_L_plot.png and ablation_Lq_plot.png.
Also prints a tabulation of all results.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

CUR_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CUR_DIR, "ablation_L_results.csv")

DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]
DATASET_LABELS = {
    "imagenet":  "ImageNet",
    "imdb_wiki": "IMDB-Wiki",
    "insta_1m":  "InstaCities-1M",
    "mirflickr": "MIR Flickr",
}

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
})


def plot_ablation(df, ablation_type, param_col, title, output_name):
    sub = df[df["ablation"] == ablation_type].copy()
    if sub.empty:
        print(f"No data for ablation={ablation_type}, skipping.")
        return

    present = [d for d in DATASETS if d in sub["dataset"].unique()]
    n = len(present)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 7 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=20, y=1.01)

    for idx, ds in enumerate(present):
        ax   = axes[idx // ncols][idx % ncols]
        rows = sub[sub["dataset"] == ds].sort_values(param_col)

        ax.plot(rows[param_col], rows["mean_query_time_ms"],
                marker="o", color="steelblue")

        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel(param_col)
        ax.set_ylabel("Mean query time (ms)")
        ax.set_xticks(rows[param_col].tolist())
        ax.grid(True, linestyle="--", alpha=0.5)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out = os.path.join(CUR_DIR, output_name)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def print_table(df):
    for ablation_type, param_col in [("L", "L"), ("Lq", "Lq")]:
        sub = df[df["ablation"] == ablation_type]
        if sub.empty:
            continue
        param_vals = sorted(sub[param_col].unique())
        print(f"\n{'─'*60}")
        print(f"  {param_col} ablation  |  mean query time (ms)")
        print(f"{'─'*60}")
        header = f"  {'Dataset':<18}" + "".join(f"{param_col}={v:>3}  " for v in param_vals)
        print(header)
        print(f"{'─'*60}")
        for ds in DATASETS:
            rows = sub[sub["dataset"] == ds]
            if rows.empty:
                continue
            row_str = f"  {DATASET_LABELS.get(ds, ds):<18}"
            for v in param_vals:
                cell = rows[rows[param_col] == v]["mean_query_time_ms"]
                row_str += f"{cell.values[0]:>8.2f}" if not cell.empty else f"{'—':>8}"
            print(row_str)
        print(f"{'─'*60}")


def main():
    if not os.path.isfile(CSV_PATH):
        print(f"CSV not found at {CSV_PATH} — run ablation_benchmark.py first.")
        return

    df = pd.read_csv(CSV_PATH)

    L_fixed  = df[df["ablation"] == "L"]["Lq"].iloc[0]
    Lq_fixed = df[df["ablation"] == "Lq"]["L"].iloc[0]

    plot_ablation(df, "L",  "L",  f"L ablation  (Lq = {L_fixed} fixed)",  "ablation_L_plot.png")
    plot_ablation(df, "Lq", "Lq", f"Lq ablation (L = {Lq_fixed} fixed)", "ablation_Lq_plot.png")
    print_table(df)


if __name__ == "__main__":
    main()