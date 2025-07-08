import torch
from torch.utils.tensorboard import SummaryWriter

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

def train(model, dataloader, criterion, optimizer, epochs, device, writer, val_loader=None):
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
