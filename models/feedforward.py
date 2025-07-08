import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, activation_fn, output_dim=10):
        super(FNN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
