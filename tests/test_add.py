from typing import Literal
import torch
import fastkern as fk
import pytest

@pytest.mark.parametrize("shape", [
    (1024, 768),    # residuals
    (1024, 3072),   # MLP up/down
    (1024, 2304),   # QKV projection bias
])

def test_add(shape: Literal[1024]) -> None:
    a = torch.randn(shape, dtype=torch.float32, device='cuda')
    b = torch.randn(shape, dtype=torch.float32, device='cuda')
    c = fk.add(a, b)
    d = a + b
    assert torch.allclose(c, d)