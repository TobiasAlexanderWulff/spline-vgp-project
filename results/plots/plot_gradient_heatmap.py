import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_layer_number(param_name: str) -> int:
    """
    Extrahiert die Layer-Nummer aus einem Parameter-Namen wie "model.3.weight".
    Falls das Format abweicht, wird 0 zurückgegeben.
    """
    parts = param_name.split(".")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return 0


def plot_heatmap(csv_path, save_dir=None, clip_min=1e-6, clip_max=1e2):
    # --- 1) CSV einlesen und Duplikate aggregieren ---
    df = pd.read_csv(csv_path)
    df = df.groupby(["epoch", "parameter"], as_index=False).mean()

    # --- 2) Split in Weights vs. Biases ---
    df_w = df[df["parameter"].str.contains("weight")].copy()
    df_b = df[df["parameter"].str.contains("bias")].copy()

    # --- 3) Pivotieren (Zeilen=Parameter, Spalten=Epoch) ---
    pivot_w = df_w.pivot(index="parameter", columns="epoch", values="mean_abs_gradient")
    pivot_b = df_b.pivot(index="parameter", columns="epoch", values="mean_abs_gradient")

    # --- 4) Sortieren nach Layer-Nummer aufsteigend ---
    sorted_w = sorted(pivot_w.index, key=extract_layer_number)
    pivot_w = pivot_w.loc[sorted_w]
    sorted_b = sorted(pivot_b.index, key=extract_layer_number)
    pivot_b = pivot_b.loc[sorted_b]

    # --- 5) Fehlende Werte auffüllen + Clipping ---
    pivot_w = pivot_w.fillna(clip_min).clip(lower=clip_min, upper=clip_max)
    pivot_b = pivot_b.fillna(clip_min).clip(lower=clip_min, upper=clip_max)

    # --- 6) Log10-Transform für Farbmapping ---
    data_w = np.log10(pivot_w.values)
    data_b = np.log10(pivot_b.values)

    # --- 7) Plot-Vorbereitung ---
    fig, (ax_w, ax_b) = plt.subplots(
        1, 2,
        figsize=(14, max(6, len(pivot_w) * 0.3)),
        sharey=True
    )
    title = Path(csv_path).parent.name.replace("_", " ").title()

    # --- 8a) Y-Achsen-Labels für Weight-Heatmap ---
    labels_w = [f"{extract_layer_number(name) // 2}" for name in pivot_w.index]

    sns.heatmap(
        data_w,
        cmap="viridis",
        vmin=-6, vmax=-1,
        cbar=False,
        ax=ax_w,
        xticklabels=pivot_w.columns,
        yticklabels=labels_w,
    )
    ax_w.set_title(f"{title} — Weight Gradients")
    ax_w.set_xlabel("Epoch")
    ax_w.set_ylabel("Layer")
    ax_w.set_yticklabels(labels_w, rotation=0)

    # --- 8b) Y-Achsen-Labels für Bias-Heatmap ---
    labels_b = [f"{extract_layer_number(name) // 2}" for name in pivot_b.index]

    sns.heatmap(
        data_b,
        cmap="viridis",
        vmin=-6, vmax=-1,
        cbar_kws={"label": "mean |Gradient|"},
        ax=ax_b,
        xticklabels=pivot_b.columns,
        yticklabels=labels_b,
    )
    ax_b.set_title(f"{title} — Bias Gradients")
    ax_b.set_xlabel("Epoch")
    ax_b.set_yticklabels(labels_b, rotation=0)

    # Farblegende hinzufügen
    cbar = ax_b.collections[0].colorbar
    cbar.set_ticks([-6, -5, -4, -3, -2, -1])
    cbar.set_ticklabels(["1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"])

    plt.tight_layout()

    # --- 9) Optional: Ergebnis speichern ---
    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(csv_path).parent.name}_gradient_heatmap.png"
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Heatmap gespeichert unter {out_path}")

    #plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Heatmap (Weights vs. Biases)")
    parser.add_argument(
        "--csv", required=True,
        help="Pfad zur gradients.csv"
    )
    parser.add_argument(
        "--save_dir", default="results/plots/",
        help="Ordner zum Speichern der Heatmap"
    )
    parser.add_argument(
        "--clip_min", type=float, default=1e-6,
        help="Untergrenze für Gradienten-Clipping vor Log"
    )
    parser.add_argument(
        "--clip_max", type=float, default=1e2,
        help="Obergrenze für Gradienten-Clipping vor Log"
    )
    args = parser.parse_args()

    plot_heatmap(
        csv_path=args.csv,
        save_dir=args.save_dir,
        clip_min=args.clip_min,
        clip_max=args.clip_max
    )