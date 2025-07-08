import torch.nn as nn
from sigmoid_spline_activation import OptimizedSigmoidSpline

def get_activation(name):
    match(name):
        case "relu":
            return nn.ReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "spline":
            return OptimizedSigmoidSpline()
        case _:
            raise ValueError(f"Activation '{name}' is not implemented")
