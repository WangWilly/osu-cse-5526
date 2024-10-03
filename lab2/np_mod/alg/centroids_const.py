from typing import Callable

import numpy as np

################################################################################

CENTROIDS_INIT_FUNC = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
