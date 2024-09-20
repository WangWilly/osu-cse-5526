from typing import Callable

import torch

################################################################################

COST_FUNC = Callable[[torch.Tensor, torch.Tensor], float]
