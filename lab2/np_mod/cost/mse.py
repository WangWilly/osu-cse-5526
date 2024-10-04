import numpy as np

from .const import Cost

################################################################################
# Mean Squared Error


def cost(y: np.ndarray, y_hat: np.ndarray) -> float:
    return np.sum((y - y_hat) ** 2) / 2.0


def cost_derivative(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    return y - y_hat


################################################################################


class MSE(Cost):
    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return cost(y, y_hat)

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return cost_derivative(y, y_hat)
