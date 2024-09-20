import torch

from .const import COST_FUNC

################################################################################
# Mean Squared Error


def cost(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    return torch.sum((y - y_hat) ** 2) / 2.0


def cost_derivative(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    return y - y_hat


cost_pair: tuple[COST_FUNC, COST_FUNC] = (cost, cost_derivative)
