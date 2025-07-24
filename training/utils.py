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
    Custom Dataset implemenation for the TinyImageNet Valdiation Data.
    This was needed because validation data is differently structured as the training data for the TinyImageNet dataset and because of this the dataloader couldnt load the valiation data.
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
    """
    Initializes the weights for a model with respect to the activation function.
    Uses xavier for sigmoid and the custom spline activation and kaiming for ReLU.
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
    """Sets all relevant randomgenerators to use a consistant seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(name: str, batch_size: int, split: str="train"):
    """Loads trainingdata and determs input_dim + output_dim."""
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
            if split == "train":
                dataset = ImageFolder(root=f"./data/tiny_imagenet-200/train", transform=transform)
            else:
                dataset = TinyImageNetValDataset(root=f"./data/tiny_imagenet-200/val", transform=transform)

        case _:
            raise ValueError(f"Unbekannter Datensatz: {name}")
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, input_dim, output_dim

