import torch.nn as nn
from models.sigmoid_spline_activation import OptimizedSigmoidSpline


def get_activation(name: str) -> nn.Module:
    """
    Returns the `nn.Module` equivalent of the activation name passed as input.
    Raises an `ValueError` if the activation is not supported.
    """
    match name:
        case "relu":
            return nn.ReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "spline":
            return OptimizedSigmoidSpline()
        case _:
            raise ValueError(f"Activation '{name}' is not implemented")
