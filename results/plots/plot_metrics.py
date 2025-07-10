import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_metrics(csv_path, save_dir=None, clip_loss=10.0, mark_outliers=True):
    df = pd.read_csv(csv_path)
    
    df = df.sort_values("epoch")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["epoch"] >= 0]
    
    experiment_name = Path(csv_path).parent.name
    
    # Loss
    plt.figure(figsize=(8, 5))
    if "train_loss" in df:
        loss_values = df["train_loss"].clip(upper=clip_loss)
        plt.plot(df["epoch"], loss_values, label="Train Loss", linewidth=2)
        if mark_outliers:
            outliers = df[df["train_loss"] > clip_loss]
            plt.scatter(outliers["epoch"], [clip_loss] * len(outliers), color="red", marker="x", label="Train Loss Outlier")

    if "val_loss" in df:
        val_loss_values = df["val_loss"].clip(upper=clip_loss)
        plt.plot(df["epoch"], val_loss_values, label="Validation Loss", linewidth=2)
        if mark_outliers:
            outliers = df[df["val_loss"] > clip_loss]
            plt.scatter(outliers["epoch"], [clip_loss] * len(outliers), color="orange", marker="x", label="Val Loss Outlier")

    plt.title(f"Loss Curve – {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / f"{experiment_name}_loss.png", dpi=300)
    #plt.show()
    
    # Accuracy
    plt.figure(figsize=(8, 5))
    if "train_accuracy" in df:
        plt.plot(df["epoch"], df["train_accuracy"], label="Train Accuracy", linewidth=2)
    if "val_accuracy" in df:
        plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy", linewidth=2)

    plt.title(f"Accuracy Curve – {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    if save_dir:
        plt.savefig(Path(save_dir) / f"{experiment_name}_accuracy.png", dpi=300)
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Pfad zur metrics.csv")
    parser.add_argument("--save_dir", type=str, default=None, help="Optionaler Speicherordner für Plots")
    parser.add_argument("--clip_loss", type=float, default=10.0, help="Maximaler Loss für Anzeige")
    parser.add_argument("--no_outliers", action="store_true", help="Keine Marker für Ausreißer anzeigen")
    args = parser.parse_args()

    plot_metrics(
        csv_path=args.csv,
        save_dir=args.save_dir,
        clip_loss=args.clip_loss,
        mark_outliers=not args.no_outliers
    )
