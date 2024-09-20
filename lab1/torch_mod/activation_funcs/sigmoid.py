import torch

################################################################################


def acti(y: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-y))


def acti_derivative(y: torch.Tensor) -> torch.Tensor:
    return acti(y) * (1.0 - acti(y))
