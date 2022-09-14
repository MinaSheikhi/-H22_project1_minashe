from re import T
from exp_decay import ExponentialDecay
from ode import solve_ode, InvalidInitialConditionError
import math
import numpy as np
import pytest


model = ExponentialDecay(0.4)

def test_ExponentialDecay():
    t = 0.0
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
              
def test_num_states():
    model = ExponentialDecay(0.4)
    model.num_states == 1

def test_solve_time():
    t0 = 0
    T = 10
    dt = 0.2

def test_solve_with_different_number_of_initial_states():
    with pytest.raises(InvalidInitialConditionError):
        solve_ode(model, u0 = [0,1], T = 10.0, dt = 0.2)


