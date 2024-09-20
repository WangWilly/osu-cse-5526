import numpy as np

from .const import Activation

################################################################################


def acti(y: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-y))


def acti_derivative(y: np.ndarray) -> np.ndarray:
    return acti(y) * (1.0 - acti(y))


################################################################################


class Sigmoid(Activation):
    def forward(self, y: np.ndarray) -> np.ndarray:
        return acti(y)

    def backward(self, y: np.ndarray) -> np.ndarray:
        return acti_derivative(y)
