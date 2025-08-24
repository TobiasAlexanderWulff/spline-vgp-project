import torch
import torch.nn.init as init
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import numpy as np
import random
import os


class TinyImageNetValDataset(Dataset):
    """
    Custom dataset implementation for the TinyImageNet validation data.
    The validation data is structured differently from the training data for the
    TinyImageNet dataset, so the default dataloader couldn't load the validation
    data.
    """
    def __init__(self, root, transform=None):
        self.transform = transform
        self.loader = default_loader
        val_dir = os.path.join(root, "images")
        annotations_path = os.path.join(root, "val_annotations.txt")
        
        # Load annotations
        with open(annotations_path, "r") as f:
            lines = f.readlines()
        
        self.img_to_class = {}
        classes = sorted({line.split()[1] for line in lines})
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        for line in lines:
            img_name, cls_name = line.split()[:2]
            self.img_to_class[img_name] = self.class_to_idx[cls_name]
        
        self.img_paths = [os.path.join(val_dir, img_name) for img_name in self.img_to_class.keys()]
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index):
        path = self.img_paths[index]
        target = self.img_to_class[os.path.basename(path)]
        img = self.loader(path)
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


def initialize_weights(model: nn.Module, activation_name: str):
    """Initialize weights based on the chosen activation function.

    Uses Xavier initialization for sigmoid or spline activations and
    Kaiming initialization for ReLU.

    Args:
        model (nn.Module): Model whose linear layers will be initialized.
        activation_name (str): Name of the activation function.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if activation_name in ("sigmoid", "spline"):
                init.xavier_uniform_(module.weight)
            else:
                init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)


def set_seed(seed: int):
    """Seed random number generators for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(name: str, batch_size: int, split: str="train"):
    """Return a DataLoader along with input and output dimensions.

    Args:
        name (str): Dataset name ("fashionmnist", "cifar10", or "tiny_imagenet").
        batch_size (int): Number of samples per batch.
        split (str, optional): Dataset split to load ("train" or "val").

    Returns:
        tuple[torch.utils.data.DataLoader, int, int]: DataLoader, input dimension,
        and number of output classes.
    """
    match(name.lower()):
        case "fashionmnist":
            input_dim = 28 * 28
            output_dim = 10
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x - x.mean()) / x.std())   # z-score per image
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
            if split == "train":
                dataset = ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform)
            else:
                dataset = TinyImageNetValDataset(root="./data/tiny-imagenet-200/val", transform=transform)

        case _:
            raise ValueError(f"Unbekannter Datensatz: {name}")
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=min(8, os.cpu_count()-2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )
    return loader, input_dim, output_dim

