import numpy as np

from .const import SIGMOID_FUNC

################################################################################


def acti_w_alpha(alpha: float) -> SIGMOID_FUNC:
    def func(y: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-alpha * y))

    return func


def acti_derivative_w_alpha(alpha: float) -> SIGMOID_FUNC:
    return lambda y: alpha * acti_w_alpha(alpha)(y) * (1.0 - acti_w_alpha(alpha)(y))


def acti_pair_w_alpha(alpha: float) -> tuple[SIGMOID_FUNC, SIGMOID_FUNC]:
    return (acti_w_alpha(alpha), acti_derivative_w_alpha(alpha))
