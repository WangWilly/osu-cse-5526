from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

################################################################################

COST_FUNC = Callable[[np.ndarray, np.ndarray], float]

################################################################################


class Cost(ABC):
    @abstractmethod
    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        pass
