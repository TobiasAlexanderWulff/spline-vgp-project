import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import math
import csv
import os
import time
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def evaluate(model, dataloader, criterion, device: str) -> tuple[float, float]:
    """Evaluate a model on a given dataloader.

    Args:
        model: Neural network to evaluate.
        dataloader: DataLoader providing evaluation data.
        criterion: Loss function used for evaluation.
        device: Device on which to run the evaluation.

    Returns:
        tuple[float, float]: Average loss and accuracy.
    """
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1) 
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = outputs.detach().argmax(dim=1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    epochs: int,
    device: str,
    val_loader=None,
    log_dir: str | None = None,
):
    """Train a model and optionally log gradient statistics.

    Args:
        model: Neural network to train.
        dataloader: Training DataLoader.
        criterion: Loss function for optimization.
        optimizer: Optimizer instance.
        epochs: Number of training epochs.
        device: Device used for training.
        val_loader: Optional validation DataLoader.
        log_dir: Directory in which to store metric logs.
    """
    gradient_log = []
    
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    
    start_time = time.time()
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0
        
        # Akkumulatoren für Gradienten über alle Batches dieser Epoche
        grad_sum_abs = defaultdict(float)
        grad_sum_norm = defaultdict(float)
        grad_count = defaultdict(int)
        
        # In run_all_experiments wird stdout in eine Log-Datei umgeleitet.
        # Damit der tqdm-Progressbar weiterhin im Terminal sichtbar ist, wird
        # seine Ausgabe explizit an das echte Terminal weitergeleitet (falls verfügbar),
        # sonst an stderr.
        try:
            real_terminal = open(os.ttyname(1), 'w')  # stdout = FD 1
        except OSError:
            real_terminal = sys.stderr
            
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
            file=real_terminal,
        )
        
        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images = images.view(images.size(0), -1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            progress_bar.set_postfix(loss=loss.item())
            loss.backward()
            
            # Gradienten pro Batch sammeln
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    grad_sum_abs[name] += grad.abs().mean().item()
                    grad_sum_norm[name] += grad.norm().item()
                    grad_count[name] += 1
            
            optimizer.step()

            total_loss += loss.item()
            predicted = outputs.detach().argmax(dim=1)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / len(dataloader.dataset)

        log_str = f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            log_str += f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"

        elapsed = int(time.time() - start_time)
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"[{hours:02}:{minutes:02}:{seconds:02}]"

        for name in grad_count.keys():
            count = max(1, grad_count[name])
            gradient_log.append({
                "epoch": epoch + 1,
                "parameter": name,
                "mean_abs_gradient": grad_sum_abs[name] / count,
                "norm_gradient": grad_sum_norm[name] / count,
                "loss_train": avg_loss,
                "loss_val": val_loss if val_loader else None,
                "acc_train": accuracy,
                "acc_val": val_acc if val_loader else None,
                "train_time": elapsed,
            })

        print(f"{time_str} {log_str}", flush=True)

    metrics_csv_path = Path(log_dir) / "metrics.csv"
    with open(metrics_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "parameter",
                "mean_abs_gradient",
                "norm_gradient",
                "loss_train",
                "loss_val",
                "acc_train",
                "acc_val",
                "train_time",
            ],
        )
        writer.writeheader()
        writer.writerows(gradient_log)
