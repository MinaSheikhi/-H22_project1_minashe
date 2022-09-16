from turtle import Pen
import numpy as np
import pytest
from exp_decay import ExponentialDecay
from ode import solve_ode, plot_ode_solution
from pendulum import Pendulum, exercise_2b

@pytest.mark.parametrize("L, theta, omega, output", [
    (1.42, np.pi/6, .35, (-9.81/1.42)*np.sin(np.pi/6)), 
    (1.42, 0, 0, 0)])

def test_pendulum(L, theta, omega, output):
    p = Pendulum(L)
    assert p(theta, omega) == output

def test_solve_pendulum_ode_with_zero_ic():
    u0 = np.array([0, 0])
    T = 10
    dt = .01
    res = exercise_2b(1, u0, T, dt, )
    
    assert res.time.all() == 0 and res.solution.all() == 0
