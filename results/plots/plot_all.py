import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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
    
    # Gruppierte Legenden-Handles
    legend_handles = [
        mpatches.Patch(color=plt.get_cmap("tab10")(0), label="Early Layers"),
        mpatches.Patch(color=plt.get_cmap("tab10")(2), label="Mid Layers"),
        mpatches.Patch(color=plt.get_cmap("tab10")(4), label="Late Layers"),
    ]    
    ax1.legend(handles=legend_handles, fontsize=10)
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
    csv_path = log_dir / "metrics.csv"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        plot_metrics(csv_path, save_dir)
    else:
        print(f"[WARN] Keine metrics.csv gefunden unter {csv_path}")

    if log_dir.exists():
        grads = extract_gradient_scalars(log_dir)
        plot_gradients(grads, args.name, save_dir, clip_min=args.clip_min)
    else:
        print(f"[WARN] Kein Log-Verzeichnis gefunden unter {log_dir}")


if __name__ == "__main__":
    main()
