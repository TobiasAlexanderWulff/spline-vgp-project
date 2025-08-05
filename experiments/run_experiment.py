import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def load_config(config_path: str) -> dict:
    """Loads a yaml config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

    from models.feedforward import FNN
    from models.activations import get_activation
    from training.trainer import train
    from training.utils import initialize_weights, set_seed, get_dataloaders

    config = load_config(args.config)
    set_seed(config["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    print(f"[INFO] Start Experiment: {config['experiment_name']}")
    
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
    
    # Weights, Loss-function and Optimizer
    initialize_weights(model, config["activation"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Training
    train(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config["epochs"],
        device=device,
        val_loader=val_loader,
        log_dir=config["log_dir"]
    )


if __name__ == "__main__":
    main()
