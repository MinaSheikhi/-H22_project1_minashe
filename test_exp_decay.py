from exp_decay import ExponentialDecay
import numpy as np
import pytest
import math

def test_ExponentialDecay():
    model = ExponentialDecay(.4)
    u = np.array([3.2])
    dudt = model(.0, u)

    assert math.isclose(dudt[-1], -1.28)

def test_ExponentialDecay_negativevalue():
    with pytest.raises(ValueError):
        model = ExponentialDecay(-3)
