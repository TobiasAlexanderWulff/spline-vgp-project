import torch
import csv
import os
import time
import sys
from pathlib import Path
from tqdm import tqdm


def evaluate(model, dataloader, criterion, device: str) -> tuple[float, float]:
    """Evaluates the model"""
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1) 
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy


def train(model, dataloader, criterion, optimizer, epochs: int, device: str, val_loader=None, log_dir: str=None):
    gradient_log = []
    start_time = time.time()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0
        try:
            real_terminal = open(os.ttyname(1), 'w')  # stdout = FD 1
        except OSError:
            real_terminal = sys.stderr
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, file=real_terminal)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            progress_bar.set_postfix(loss=loss.item())
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
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
        
                # Metriken loggen
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.abs().mean()
                grad_norm = param.grad.norm()
                gradient_log.append({
                    "epoch": epoch+1,
                    "parameter": name,
                    "mean_abs_gradient": grad_mean.item(),
                    "norm_gradient": grad_norm.item(),
                    "loss_train": avg_loss,
                    "loss_val": val_loss if val_loader else None,
                    "acc_train": accuracy,
                    "acc_val": val_acc if val_loader else None,
                    "train_time": elapsed,
                })
        
        print(f"{time_str} {log_str}", flush=True)

    metrics_csv_path = Path(log_dir) / "metrics.csv"
    with open(metrics_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", 
            "parameter", 
            "mean_abs_gradient", 
            "norm_gradient", 
            "loss_train", 
            "loss_val",
            "acc_train",
            "acc_val",
            "train_time"
        ])
        writer.writeheader()
        writer.writerows(gradient_log)
