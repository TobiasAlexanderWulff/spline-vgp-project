import sys
from pathlib import Path
import argparse
import yaml
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Projekt-Stammverzeichnis zum Modulpfad hinzufügen
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.feedforward import FNN
from models.activations import get_activation
from training.trainer import train, create_csv_logger
from training.utils import set_seed, get_dataloaders

import os

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    print(f"[INFO] Starte Experiment: {config['experiment_name']}")
    
    # Dataset & Input Dimension
    train_loader, input_dim, output_dim = get_dataloaders(config["dataset"], config["batch_size"])
    val_loader, _, _ = get_dataloaders(config["dataset"], config["batch_size"], split="val")
    
    # Modell & Aktivierung
    activation_fn = get_activation(config["activation"])
    model = FNN(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        depth=config["depth"],
        activation_fn=activation_fn,
        output_dim=output_dim,
    )
    
    # Loss, Optimizer, Writer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    writer = SummaryWriter(log_dir=config["log_dir"])
    csv_path = create_csv_logger(config["log_dir"])

    start_time = time.time()

    # Training
    train(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config["epochs"],
        device=device,
        writer=writer,
        val_loader=val_loader,
        csv_path=csv_path
    )
    
    duration = time.time() - start_time
    print(f"[INFO] Training duration: {duration:.2f} seconds")
    
    log_path = Path(config["log_dir"]) / "training_duration.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"{duration:.2f} seconds\n")
    
    writer.close()
    
    # Zusammenfassung laden
    final_epoch = config["epochs"]
    summary = {
        "experiment": config["experiment_name"],
        "duration": duration,
        "epoch": final_epoch,
        "train_loss": None,
        "train_acc": None,
        "val_loss": None,
        "val_acc": None,
    }
    
    # Werte aus CSV holen
    csv_path = Path(config["log_dir"]) / "metrics.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            last_row = df.iloc[-1]
            summary["train_loss"] = last_row["train_loss"]
            summary["train_acc"] = last_row["train_accuracy"]
            summary["val_loss"] = last_row.get("val_loss", None)
            summary["val_acc"] = last_row.get("val_accuracy", None)

    # Zeit formatieren
    duration_min = int(duration // 60)
    duration_sec = int(duration % 60)
    duration_str = f"{duration_min:02}:{duration_sec:02}"
    
    # Konsolen Output
    print("\n📋 Training Summary (" + summary["experiment"] + ")")
    print("─" * 46)
    print(f"Total Time      : {duration_str}")
    print(f"Final Epoch     : {summary['epoch']}")
    print(f"Train Loss      : {summary['train_loss']:.4f}")
    print(f"Train Accuracy  : {summary['train_acc']:.4f}")
    if summary["val_loss"] is not None and not pd.isna(summary["val_loss"]):
        print(f"Val Loss        : {summary['val_loss']:.4f}")
        print(f"Val Accuracy    : {summary['val_acc']:.4f}")
    print("─" * 46)
    

if __name__ == "__main__":
    main()
