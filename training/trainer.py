import torch
from torch.utils.tensorboard import SummaryWriter

import csv
import os
import time

def create_csv_logger(log_dir, filename="metrics.csv"):
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, filename)
    
    # Header schreiben (falls die Datei neu ist)
    if not os.path.exists(filepath):
        with open(filepath, mode="w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "timestamp"])
    
    return filepath

def evaluate(model,dataloader, criterion, device):
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

def train(model, dataloader, criterion, optimizer, epochs, device, writer, val_loader=None, csv_path=None):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradienten loggen
            for name, param in model.named_parameters():
                if param.name is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, epoch)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / len(dataloader.dataset)
        
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        
        log_str = f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            log_str += f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"

        print(log_str)
        
        with open (csv_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch + 1,
                avg_loss,
                accuracy,
                val_loss if val_loader else "",
                val_acc if val_loader else "",
                time.strftime("%Y-%m-%d %H:%M:%S")
            ])
