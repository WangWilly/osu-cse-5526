import os
import sys

import torch

this_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(this_dir, "../..")))

torch.manual_seed(8787)
