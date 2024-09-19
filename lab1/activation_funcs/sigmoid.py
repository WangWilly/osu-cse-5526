from typing import Callable

import numpy as np

################################################################################


def acti(y: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-y))


def acti_derivative(y: np.ndarray) -> np.ndarray:
    return acti(y) * (1.0 - acti(y))


acti_pair: Callable[[np.ndarray], np.ndarray] = (acti, acti_derivative)
