# experiment/run_experiment.py

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
from training.trainer import train
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
    
    # Dataset & Input Dimension
    train_loader, input_dim, output_dim = get_dataloaders(config["dataset"], config["batch_size"])
    
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

    start_time = time.time()

    # Training
    train(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config["epochs"],
        device=device,
        writer=writer
    )
    
    duration = time.time() - start_time
    print(f"[INFO] Training duration: {duration:.2f} seconds")
    
    log_path = Path(config["log_dir"]) / "training_duration.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"{duration:.2f} seconds\n")
    
    writer.close()

if __name__ == "__main__":
    main()
