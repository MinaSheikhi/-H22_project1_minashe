from exp_decay import ExponentialDecay
import numpy as np
import pytest
import math
from ode import ODEModel, solve_ode
from exception import InvalidInitialConditionError

model = ExponentialDecay(.4)

def test_ExponentialDecay():
    u = np.array([3.2])
    dudt = model(3, u)
    assert math.isclose(dudt[-1], -1.28)

def test_negative_decay_raises_ValueError_1a():
    with pytest.raises(ValueError):
        model = ExponentialDecay(-3)

def test_negative_decay_raises_ValueError_1b():
    with pytest.raises(ValueError):
        model.decay = -1.0

def test_num_states():
    assert model.num_states == 1

def test_solve_with_different_number_of_initial_states():
    u0 = np.array([1, 2])
    with pytest.raises(InvalidInitialConditionError):
        solve_ode(model, u0, 10, .1)

#@pytest.mark.parametrize("a, u0, T, dt", ([], [], []))
# def test_solve_time(a, u0, T, dt):
#     pass

# def test_solve_solution(a, u0, T, dt):
#     t = np.arange(0, T, dt)
    #assert ExponentialDecay(a).solve_exponentional_decay() == u0*np.exp(-a*t)


test_ExponentialDecay()
