import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_heatmap(csv_path, save_dir=None, clip_min=1e-6, clip_max=1e2):
    df = pd.read_csv(csv_path)
    df = df.groupby(["epoch", "parameter"], as_index=False).mean()
    
    pivot = df.pivot(index="parameter", columns="epoch", values="mean_abs_gradient")
    
    pivot = pivot.fillna(clip_min)
    
    clipped = np.clip(pivot.values, clip_min, clip_max)
    data_log = np.log10(clipped)
    
    # Heatmap zeichnen
    plt.figure(figsize=(12, max(6, len(pivot) * 0.4)))
    ax = sns.heatmap(data_log, cmap="viridis", cbar_kws={"label": "log₁₀(mean |Gradient|)"})
    ax.set_title(Path(csv_path).parent.name + " - Gradient Heatmap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer / Parameter")
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / f"{Path(csv_path).parent.name}_gradient_heatmap.png"
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Heatmap gespeichert unter {out_path}")
    
    plt.tight_layout()
    #plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Pfad zur gradients.csv")
    parser.add_argument("--save_dir", default="results/plots/", help="Speicherort für Heatmap")
    parser.add_argument("--clip_min", type=float, default=1e-6, help="Min. Gradient für Log-Darstellung")
    parser.add_argument("--clip_max", type=float, default=1e2, help="Max. Gradient für Log-Darstellung")
    args = parser.parse_args()
    
    plot_heatmap(
        csv_path=args.csv,
        save_dir=args.save_dir,
        clip_min=args.clip_min,
        clip_max=args.clip_max
    )

