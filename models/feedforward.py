import torch
import torch.nn as nn


class FFN(nn.Module):
    """Feed Forward Network"""
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, activation_fn: nn.Module, output_dim: int):
        super(FFN, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
        
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass a tensor through the network."""
        return self.model(x)
