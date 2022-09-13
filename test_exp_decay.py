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

