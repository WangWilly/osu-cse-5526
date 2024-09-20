from typing import Callable

import torch

################################################################################

SIGMOID_FUNC = Callable[[torch.Tensor], torch.Tensor]
