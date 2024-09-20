import torch

################################################################################
# Mean Squared Error


def cost(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    return torch.sum((y - y_hat) ** 2) / 2.0


def cost_derivative(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    return y - y_hat
