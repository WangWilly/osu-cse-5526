import numpy as np
from typing import Callable

################################################################################

CENTROIDS_INIT_FUNC = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
