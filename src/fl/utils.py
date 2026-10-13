import os
import random
import torch
import numpy as np
from collections import OrderedDict
from typing import List

def set_seed(seed):
    """Seed Python, NumPy and PyTorch and make repeated runs give identical results.

    Without the deterministic settings, two GPU runs with the same seed drift
    apart after a few rounds. Does nothing if `seed` is None.
    """
    if seed is None:
        return
    # cuBLAS only behaves deterministically with this set, and reads it on first use.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    """Set model parameters from a list of numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
