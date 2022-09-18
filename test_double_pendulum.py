import numpy as np
from ode import solve_ode
import pytest

from double_pendulum import DoublePendulum, solve_double_pendulum
from ode import solve_ode

model = DoublePendulum()

def test_derivatives_at_rest_is_zero():
    u = np.array([0, 0, 0, 0])
    dudt = model(0, u)

    assert dudt.all() == 0

@pytest.mark.parametrize(
    "theta1, theta2, expected", [
    (0, 0, 0),
    (0, 0.5, 3.386187037),
    (0.5, 0, -7.678514423),
    (0.5, 0.5, -4.703164534)])

def test_domega1_dt(theta1, theta2, expected):
    model = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    dtheta1_dt, domega1_dt, _, _ = model(t, y)

    assert np.isclose(dtheta1_dt, 0.25)
    #assert np.isclose(domega1_dt, expected)

@pytest.mark.parametrize(
    "theta1, theta2, expected", [
    (0, 0, 0.0),
    (0, 0.5, -7.704787325),
    (0.5, 0, 6.768494455),
    (0.5, 0.5, 0.0)])

def test_domega2_dt(theta1, theta2, expected):
    model = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    _, _, dtheta2_dt, domega2_dt = model(t, y)

    assert np.isclose(dtheta2_dt, 0.15)
    #assert np.isclose(domega2_dt, expected)

def test_solve_pendulum_ode_with_zero_ic():
    res  = solve_ode(DoublePendulum(), np.array([0,0,0,0]), 10, .1)

    assert res.solution.all() == 0

def test_solve_double_pendulum_function_zero_ic():
    dp = DoublePendulum()
    res = solve_double_pendulum(np.array([0, 0, 0, 0]), 10, pendulum=dp)

    assert res.theta1.all() == 0
    assert res.omega1.all() == 0
    assert res.theta2.all() == 0
    assert res.omega2.all() == 0
    assert res.x1.all() == 0
    assert res.x2.all() == 0
    #assert res.y1.all() == -dp.L1
    #assert res.y2.any() == -(dp.L1 + dp.L2)

test_solve_double_pendulum_function_zero_ic()