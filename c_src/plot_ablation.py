#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Define mappings for labels and datasets
ALGO_MAPPING = {
    "single": "K-btnn (serial)",
    "single_threaded": "K-btnn (T = 16, threaded)",
    "single_parallel": "K-btnn (T = 16, parallel)",
    "double": "K-dgtnn (serial)",
    "double_threaded": "K-dgtnn (T = 16)"
}

DATASET_MAPPING = {
    "imagenet": "Dataset: ImageNet",
    "imdb_wiki": "Dataset: IMDB wiki",
    "insta_1m": "Dataset: InstaCities-1M",
    "mirflickr": "Datset: MIR Flickr"
}

DATA_TYPES = {
    'N': 'Dataset Size (N)',
    'K': 'Number of Neighbors (K)'
}

# Load data
csv_path = os.path.join(CUR_DIR, "ablation_results.csv")
df = pd.read_csv(csv_path)

# Set global plot parameters for better visibility in papers
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 30,
    'lines.linewidth': 3,
    'lines.markersize': 10
})

def plot_ablation(data_type, output_name, title):
    # Filter for N or K
    sub_df = df[df['N/K'].str.startswith(data_type)].copy()
    verbose_data_type = DATA_TYPES['N' if data_type.startswith('N') else 'K']
    sub_df['Value'] = sub_df['N/K'].str.extract('(\d+)').astype(int)
    
    datasets = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]
    algos = ["single", "single_threaded", "single_parallel", "double", "double_threaded"]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    # fig.suptitle(title, y=1.03)
    
    for i, dset in enumerate(datasets):
        ax = axes[i // 2, i % 2]
        dset_df = sub_df[sub_df['Dataset'] == dset]
        
        for algo in algos:
            algo_df = dset_df[dset_df['Algorithm'] == algo].sort_values('Value')
            if not algo_df.empty:
                ax.plot(algo_df['Value'], algo_df['Time (ms)'], marker='o', label=ALGO_MAPPING.get(algo, algo))
        
        ax.set_title(DATASET_MAPPING.get(dset, dset))
        ax.set_xlabel(verbose_data_type)
        ax.set_ylabel("Time (ms)")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(frameon=True, shadow=True, markerscale=1.5)
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join("c_src", output_name), bbox_inches='tight', dpi=300)
    print(f"Saved {output_name}")

if __name__ == "__main__":
    plot_ablation('N', 'ablation_N.png', 'Ablation Study: Scalability with Dataset Size (N)')
    plot_ablation('K', 'ablation_K.png', 'Ablation Study: Performance vs. Number of Neighbors (K)')
