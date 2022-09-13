from exp_decay import ExponentialDecay
import math
import numpy as np
import pytest

def test_ExponentialDecay():
    t = 0.0
    model = ExponentialDecay(0.4)
    u = np.array([3.2])
    du_dt = model(t,u)
    assert math.isclose(du_dt[0],-1.28)


def test_negative_decay_raises_ValueError_1a():
    with pytest.raises(ValueError):
        model = ExponentialDecay(-0.4)

def test_negative_decay_raises_ValueError_1b():
    model = ExponentialDecay(0.4) #set a = 0.4
    with pytest.raises(ValueError):
        model.decay = -1.0  # cant set a to new value -1 because of valueerror
              


