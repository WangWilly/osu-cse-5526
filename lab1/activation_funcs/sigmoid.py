from typing import Callable
import numpy as np

################################################################################

SIGMOID_FUNC = Callable[[np.ndarray], np.ndarray]

################################################################################

def sigmoid(y: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-y))

def sigmoid_derivative(y: np.ndarray) -> np.ndarray:
    return sigmoid(y) * (1 - sigmoid(y))

################################################################################

def sigmoid_w_alpha(alpha: float) -> SIGMOID_FUNC:
    def func(y: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-alpha * y))
    return func

def sigmoid_derivative_w_alpha(alpha: float) -> SIGMOID_FUNC:
    return lambda y: alpha * sigmoid_w_alpha(alpha)(y) * (1 - sigmoid_w_alpha(alpha)(y))
