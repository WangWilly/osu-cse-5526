from typing import Callable
import numpy as np

################################################################################

SIGMOID_FUNC = Callable[[np.ndarray], np.ndarray]

################################################################################

def sigmoid(y: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-y))

def sigmoid_derivative(y: np.ndarray) -> np.ndarray:
    return sigmoid(y) * (1.0 - sigmoid(y))

sigmoid_pair: Callable[[np.ndarray], np.ndarray] = (sigmoid, sigmoid_derivative)

################################################################################

def sigmoid_w_alpha(alpha: float) -> SIGMOID_FUNC:
    def func(y: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-alpha * y))
    return func

def sigmoid_derivative_w_alpha(alpha: float) -> SIGMOID_FUNC:
    return lambda y: alpha * sigmoid_w_alpha(alpha)(y) * (1.0 - sigmoid_w_alpha(alpha)(y))
