"""V6 compatibility for RF-DETR's optional batched assignment extension."""
from __future__ import annotations

import importlib.util

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def solve_to_indices_scipy(stacked: torch.Tensor, num_matches: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a uniform batch with SciPy and return RF-DETR-compatible CPU indices."""
    if stacked.ndim != 3:
        raise ValueError(f"expected a 3D cost tensor, got shape {tuple(stacked.shape)}")
    expected = min(stacked.shape[1], stacked.shape[2])
    if num_matches != expected:
        raise ValueError(f"num_matches={num_matches} does not match matrix shape {tuple(stacked.shape[1:])}")
    if expected == 0:
        empty = torch.empty((stacked.shape[0], 0), dtype=torch.int64)
        return empty, empty.clone()

    costs = stacked.detach().float().cpu().numpy()
    pairs = [linear_sum_assignment(cost) for cost in costs]
    rows = torch.from_numpy(np.stack([pair[0] for pair in pairs])).to(torch.int64)
    cols = torch.from_numpy(np.stack([pair[1] for pair in pairs])).to(torch.int64)
    return rows, cols


def install_assignment_fallback() -> str:
    """Use SciPy only when RF-DETR's optional native assignment package is absent."""
    if importlib.util.find_spec("torch_linear_assignment") is not None:
        return "torch_linear_assignment"
    from rfdetr.models import _assignment

    _assignment._solve_to_indices = solve_to_indices_scipy
    return "scipy"
