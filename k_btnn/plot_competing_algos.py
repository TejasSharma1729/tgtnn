#!/usr/bin/env python3
"""QPS vs Recall@10 comparison — 300K datapoints, Phase 1 Build."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "competing_algos_ablation_results.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "competing_algos_comparison.png")

df = pd.read_csv(CSV)
df = df[(df["Phase"] == 1) & (df["Action"] == "Build")]

OUR_ALGOS = {
    "KNNS Binary (Threaded)": ("*",  200),
    "KNNS Binary (Serial)":   ("P",  120),
    "KNNS Double (Threaded)": ("D",  100),
    "KNNS Double (Serial)":   ("^",  100),
}

COMPETING_GROUPS = {
    "Linscan":   ["Linscan (budget: 1)", "Linscan (budget: 10)", "Linscan (budget: 100)",
                  "Linscan (budget: 1000)", "Linscan (base)"],
    "CuFe":      ["Cufe (budget: 1)", "Cufe (budget: 10)", "Cufe (budget: 100)",
                  "Cufe (budget: 1000)", "Cufe (base)"],
    "SHNSW":     ["SHNSW (ef_search: 40)", "SHNSW (ef_search: 200)",
                  "SHNSW (ef_search: 500)", "SHNSW (ef_search: 1000)"],
    "Faiss HNSW":["Faiss HNSW (efSearch: 64)", "Faiss HNSW (efSearch: 128)",
                  "Faiss HNSW (efSearch: 256)"],
    "Faiss GT":  ["Faiss GT"],
    "ScaNN":     ["Scann"],
    "Falconn":   ["Falconn (num_tables: 50)", "Falconn (num_tables: 100)"],
}

COMP_COLORS = {
    "Linscan":    "#e41a1c",
    "CuFe":       "#ff7f00",
    "SHNSW":      "#984ea3",
    "Faiss HNSW": "#4daf4a",
    "Faiss GT":   "#a65628",
    "ScaNN":      "#999999",
    "Falconn":    "#f781bf",
}
OUR_COLOR = "#1f78b4"

DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]
TITLES   = {"imagenet": "ImageNet", "imdb_wiki": "IMDb-Wiki",
            "insta_1m": "Instagram-1M", "mirflickr": "MIRFlickr"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, dataset in zip(axes, DATASETS):
    sub = df[df["Dataset"] == dataset]

    # --- competing algorithms ---
    for grp_name, algos in COMPETING_GROUPS.items():
        rows = sub[sub["Algo"].isin(algos)].sort_values("Recall")
        if rows.empty:
            continue
        lw   = 1.0 if len(rows) == 1 else 1.5
        mkr  = "o"
        ax.plot(rows["Recall"], rows["QPS"],
                marker=mkr, color=COMP_COLORS[grp_name], label=grp_name,
                linewidth=lw, markersize=5, alpha=0.85)

    # --- our algorithms (all at recall=1.0, stacked vertically) ---
    for algo_name, (mkr, sz) in OUR_ALGOS.items():
        row = sub[sub["Algo"] == algo_name]
        if row.empty:
            continue
        ax.scatter(row["Recall"], row["QPS"],
                   color=OUR_COLOR, marker=mkr, s=sz, zorder=6,
                   edgecolors="black", linewidths=0.5,
                   label=algo_name)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.1f}"))
    ax.set_xlim(-0.02, 1.08)
    ax.set_title(TITLES[dataset], fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS  (log scale)", fontsize=11)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=9)

# --- shared legend (bottom centre) ---
handles, labels = axes[0].get_legend_handles_labels()
# collect from all axes to catch any algo that only appears in later datasets
seen = set(labels)
for ax in axes[1:]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
            handles.append(h); labels.append(l); seen.add(l)

fig.legend(handles, labels, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.03), fontsize=9,
           frameon=True, framealpha=0.9)

fig.suptitle("QPS vs Recall@10  —  300 K datapoints  (Phase 1 Build)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
