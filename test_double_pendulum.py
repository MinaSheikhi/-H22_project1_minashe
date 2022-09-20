import numpy as np
import pytest

from double_pendulum import DoublePendulum, solve_double_pendulum
from ode import solve_ode

# Oppgave 3b
model = DoublePendulum()

def test_derivatives_at_rest_is_zero():
    t = 0
    u0 = np.array([0, 0, 0, 0])
    
    assert all(model(t, u0) == 0)

@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0),
        (0, 0.5, 3.386187037),
        (0.5, 0, -7.678514423),
        (0.5, 0.5, -4.703164534),
    ],
)
def test_domega1_dt(theta1, theta2, expected):
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    dtheta1_dt, domega1_dt, _, _ = model(t, y)
    assert np.isclose(dtheta1_dt, 0.25)
    assert np.isclose(domega1_dt, expected)


@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0.0),
        (0, 0.5, -7.704787325),
        (0.5, 0, 6.768494455),
        (0.5, 0.5, 0.0),
    ],
)
def test_domega2_dt(theta1, theta2, expected):
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    _, _, dtheta2_dt, domega2_dt = model(t, y)
    assert np.isclose(dtheta2_dt, 0.15)
    assert np.isclose(domega2_dt, expected)


def test_solve_pendulum_ode_with_zero_ic():
    t = 0
    u0 = np.array([0, 0, 0, 0])

    solutions =  solve_ode(model = model, u0 = u0, T = 10, dt = 0.01)
    time, result = solutions

    assert all(result[0] == 0) 
    assert all(result[1] == 0) 
    assert all(result[2] == 0) 
    assert all(result[3] == 0)

def test_solve_double_pendulum_function_zero_ic():
    results = solve_double_pendulum(u0 = np.array([0, 0, 0, 0]), T = 10, dt = 0.1)
          
    assert all(results.theta1 == 0)
    assert all(results.omega1 == 0)
    assert all(results.theta2 == 0)
    assert all(results.omega2 == 0)
    assert all(results.x1 == 0)
    assert all(results.x2 == 0)
    assert all(results.y1 == -model.L1)
    assert all(results.y2 == -(model.L1 + model.L2))

