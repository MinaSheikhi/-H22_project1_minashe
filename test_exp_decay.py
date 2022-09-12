from exp_decay import ExponentialDecay
import numpy as np
import pytest
import math

def test_ExponentialDecay():
    model = ExponentialDecay(.4)
    u = np.array([3.2])
    dudt = model(.0, u)

    assert math.isclose(dudt[-1], -1.28)

def test_negative_decay_raises_ValueError_1a():
    with pytest.raises(ValueError):
        model = ExponentialDecay(-3)

def test_negative_decay_raises_ValueError_1b():
    model = ExponentialDecay(0.4)
    
    with pytest.raises(ValueError):
        model.decay = -1.0