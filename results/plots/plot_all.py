import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns


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


def plot_heatmap(csv_path, save_dir=None, clip_min=1e-6, clip_max=1e0):
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


def plot_metrics(csv_path, save_dir, clip_loss=10.0, mark_outliers=True):
    df = pd.read_csv(csv_path)
    df = df.sort_values("epoch")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["epoch"] >= 0]

    name = Path(csv_path).parent.name

    # --- LOSS ---
    plt.figure(figsize=(8, 5))
    if "train_loss" in df:
        plt.plot(df["epoch"], df["train_loss"].clip(upper=clip_loss), label="Train Loss")
        if mark_outliers:
            out = df[df["train_loss"] > clip_loss]
            plt.scatter(out["epoch"], [clip_loss]*len(out), color="red", marker="x")
    if "val_loss" in df:
        plt.plot(df["epoch"], df["val_loss"].clip(upper=clip_loss), label="Val Loss")
        if mark_outliers:
            out = df[df["val_loss"] > clip_loss]
            plt.scatter(out["epoch"], [clip_loss]*len(out), color="orange", marker="x")

    plt.title(f"Loss – {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(Path(save_dir) / f"{name}_loss.png", dpi=300)
    plt.close()

    # --- ACCURACY ---
    plt.figure(figsize=(8, 5))
    if "train_accuracy" in df:
        plt.plot(df["epoch"], df["train_accuracy"], label="Train Acc")
    if "val_accuracy" in df:
        plt.plot(df["epoch"], df["val_accuracy"], label="Val Acc")
    plt.title(f"Accuracy – {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(Path(save_dir) / f"{name}_accuracy.png", dpi=300)
    plt.close()


def extract_gradient_scalars(log_dir):
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    tags = [t for t in ea.Tags()["scalars"] if t.startswith("gradients/norm_")]
    grads = {"weight": {}, "bias": {}}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        name = tag.split("/")[-1]
        key = "weight" if "weight" in name else "bias"
        grads[key][name] = (steps, vals)
    return grads


def plot_gradients(gradient_data, experiment_name, save_dir, outlier_threshold=1000.0, clip_min=1e-6):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Mean |Gradient| (log) – {experiment_name}")

    for key, ax in zip(("weight", "bias"), (ax1, ax2)):
        color_map = plt.get_cmap("tab10")
        for idx, (name, (steps, values)) in enumerate(gradient_data[key].items()):
            steps = np.array(steps)
            values = np.clip(values, clip_min, None)

            # Layer-Index extrahieren
            try:
                layer_num = int(name.split(".")[1])
            except:
                layer_num = 0  # fallback

            # Farbgruppe zuweisen
            if layer_num < 6:
                color = color_map(0)
            elif layer_num < 12:
                color = color_map(2)
            else:
                color = color_map(4)

            ax.plot(steps, values, label=name, color=color)

            # Optional: Marker für Ausreißer
            outliers = np.array(steps)[np.array(values) > outlier_threshold]
            if len(outliers) > 0:
                ax.scatter(outliers, [outlier_threshold]*len(outliers), color=color, s=20, marker="x")

        ax.set_yscale("log")
        ax.set_ylabel(f"{key.capitalize()} Gradients")
        ax.grid(True)
        if ax.has_data():
            ax.legend(fontsize=8, ncol=2)
    ax2.set_xlabel("Epoch")
    plt.tight_layout()
    out_path = Path(save_dir) / f"{experiment_name}_scalar_gradients.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment-Name (entspricht Verzeichnis unter logs/tensorboard/)")
    parser.add_argument("--save_dir", default="results/plots", help="Pfad für gespeicherte Plots")
    parser.add_argument("--clip_min", type=float, default=1e-6, help="Untere Grenze für log y-Achse (Gradienten)")
    args = parser.parse_args()

    log_dir = Path(f"logs/tensorboard/{args.name}")
    csv_path_metrics = log_dir / "metrics.csv"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if csv_path_metrics.exists():
        plot_metrics(csv_path_metrics, save_dir)
    else:
        print(f"[WARN] Keine metrics.csv gefunden unter {csv_path_metrics}")

    if log_dir.exists():
        grads = extract_gradient_scalars(log_dir)
        plot_gradients(grads, args.name, save_dir, clip_min=args.clip_min)
    else:
        print(f"[WARN] Kein Log-Verzeichnis gefunden unter {log_dir}")
    
    csv_path_heatmap = log_dir / "gradients.csv"
    if csv_path_heatmap.exists():
        plot_heatmap(
            csv_path=csv_path_heatmap,
            save_dir=str(Path(args.save_dir)),
            clip_min=1e-6,
            clip_max=1e0,
        )


if __name__ == "__main__":
    main()
