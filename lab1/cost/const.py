from typing import Callable

import numpy as np

################################################################################

COST_FUNC = Callable[[np.ndarray, np.ndarray], float]
