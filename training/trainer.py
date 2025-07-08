import torch
from torch.utils.tensorboard import SummaryWriter

def train(model, dataloader, criterion, optimizer, epochs, device, writer):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Logging gradients
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
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
