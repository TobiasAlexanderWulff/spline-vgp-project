import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(csv_path, save_dir=None):
    df = pd.read_csv(csv_path)
    
    experiment_name = Path(csv_path).parent.name
    
    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", linewidth=2)
    plt.title(f"Loss Curve - {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / f"{experiment_name}_loss.png", dpi=300)
    plt.show()
    
    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_accuracy"], label="Train Accuracy", linewidth=2)
    if "val_accuracy" in df.columns and df["val_accuracy"].notna().any():
        plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy", linewidth=2)
    plt.title(f"Accuracy Curve – {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    if save_dir:
        plt.savefig(Path(save_dir) / f"{experiment_name}_accuracy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Pfad zur metrics.csv")
    parser.add_argument("--save_dir", type=str, default=None, help="Optionaler Speicherordner für Plots")
    args = parser.parse_args()
    
    plot_metrics(args.csv, save_dir=args.save_dir)
