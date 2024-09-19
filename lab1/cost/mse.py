from typing import Callable
import numpy as np

################################################################################

COST_FUNC = Callable[[np.ndarray, np.ndarray], float]

################################################################################
# Mean Squared Error

def cost(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    return np.sum((Y - Y_hat) ** 2) / 2.0
