import torch
import torch.nn.init as init
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from models.activations import OptimizedSigmoidSpline
import numpy as np
import random


def initialize_weights(model, activation_name):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if activation_name in ("sigmoid", "spline"):
                init.xavier_uniform_(module.weight)
            else:
                init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)


def set_seed(seed):
    """Setzt alle relevanten Zufallsgeneratoren auf einen festen Seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(name, batch_size, split="train"):
    """Lädt Trainingsdaten und setzt input_dim + output_dim."""

    match(name.lower()):
        case "fashionmnist":
            input_dim = 28 * 28
            output_dim = 10
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x - x.mean()) / x.std())   # z-Score pro Bild
            ])
            dataset = torchvision.datasets.FashionMNIST(
                root="./data", train=(split=="train"), download=True, transform=transform)

        case "cifar10":
            input_dim = 3 * 32 * 32
            output_dim = 10
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
            ])
            dataset = torchvision.datasets.CIFAR10(
                root="./data/cifar10", train=(split=="train"), download=True, transform=transform)

        case "tiny_imagenet":
            input_dim = 3 * 64 * 64
            output_dim = 200
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            subfolder = "train" if split == "train" else "val"
            dataset = ImageFolder(root=f"./data/tiny_imagenet-200/{subfolder}", transform=transform)

        case _:
            raise ValueError(f"Unbekannter Datensatz: {name}")
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, input_dim, output_dim

