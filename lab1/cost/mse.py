import numpy as np

from .const import COST_FUNC

################################################################################
# Mean Squared Error


def cost(y: np.ndarray, y_hat: np.ndarray) -> float:
    return np.sum((y - y_hat) ** 2) / 2.0


def cost_derivative(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    return y - y_hat


cost_pair: tuple[COST_FUNC, COST_FUNC] = (cost, cost_derivative)
