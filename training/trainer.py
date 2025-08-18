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
from diagnostics import register_spline_outside_hooks, check_and_log_anomaly
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
    
    diag_dir = Path(log_dir) / "diag"
    anomalies_csv = diag_dir / "anomalies.csv"
    outside_csv   = diag_dir / "outside.csv"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images = images.view(images.size(0), -1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            progress_bar.set_postfix(loss=loss.item())
            loss.backward()
            
            # Gesamtnorm der Gradienten (ohne Clipping)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')).item()
            lr = optimizer.param_groups[0]["lr"]

            # Anomalien loggen (NaN/Inf oder sehr große Norm)
            check_and_log_anomaly(
                epoch=epoch + 1,
                phase="train",
                batch_idx=batch_idx,   # oder bidx – je nach deinem Variablennamen
                loss=loss.item(),
                grad_norm=gnorm,
                lr=lr,
                csv_path=str(anomalies_csv),
            )
            if (not math.isfinite(gnorm)) or gnorm > 1e3 or (not math.isfinite(loss.item())):
                print(f"[DIAG] epoch={epoch+1} batch={batch_idx} loss={loss.item():.3g} gnorm={gnorm:.3g} lr={lr:.2e}")
            
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

        # Einen Val-Batch ziehen (deterministisch genug für Trend)
        val_images, _val_labels = next(iter(val_loader))
        val_images = val_images.to(device, non_blocking=True)
        
        val_images = val_images.view(val_images.size(0), -1)

        # Hooks registrieren -> einmal forward -> Hooks wieder entfernen
        handles, stats = register_spline_outside_hooks(model)
        model.eval()
        with torch.no_grad():
            _ = model(val_images)
        for h in handles:
            h.remove()

        # Pro Layer die mittlere Outside-Quote in CSV schreiben
        stats.dump_csv(str(outside_csv), epoch=epoch+1)

        elapsed = int(time.time() - start_time)
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"[{hours:02}:{minutes:02}:{seconds:02}]"

        # Mittel der Parameter pro Epoche loggen
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
