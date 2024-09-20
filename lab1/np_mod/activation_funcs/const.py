from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

################################################################################

SIGMOID_FUNC = Callable[[np.ndarray], np.ndarray]

################################################################################


class Activation(ABC):
    @abstractmethod
    def forward(self, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, y: np.ndarray) -> np.ndarray:
        pass
