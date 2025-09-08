import torch
import fastkern as fk
import pytest
from typing import Literal

@pytest.mark.parametrize("shape", [
    # (1024, 1600, 6400),   # MLP up-projection
    # (1024, 6400, 1600),   # MLP down-projection
    # (1024, 64, 1024)      # Attention scoring
    (1024, 256, 1024)
])

def test_mult(shape):
    a = torch.randn(shape[0:2], dtype=torch.float32, device='cuda')
    b = torch.randn(shape[1:], dtype=torch.float32, device='cuda')
    c = fk.mult(a, b)
    d = a @ b
    assert torch.allclose(c, d)