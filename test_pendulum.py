import numpy as np
import pytest
from pendulum import Pendulum, exercise_2b, solve_pendulum


@pytest.mark.parametrize("L, theta, omega, dOmega", [
    (1.42, np.pi/6, 0.35, (-9.81/1.42)*np.sin(np.pi/6)), 
    (1.42, 0, 0, 0)])

def test_Pendulum(L, omega, theta, dOmega):
    p = Pendulum(L = L)
    u = np.array([theta, omega])

    assert p(0, u)[0] == omega and p(0,u)[1] == dOmega

    """
    derivatives = p(theta= theta, omega= omega)
    theta_comp = derivatives[0]
    omega_comp = derivatives[1]
    
    assert theta_comp == omega and omega_comp == dOmega
"""

def test_solve_pendulum_ode_with_zero_ic():
    u0 = np.array([0, 0])
    res = exercise_2b(u0, T = 10, dt = 0.01)
    
    print(res.result)

    assert all(res.result[0] == 0) and all(res.restult[1] ==  0)
"""
def test_solve_pendulum_function_zero_ic():
    solution = solve_pendulum(u0 = np.array([0, 0]), T = 10, dt = 0.1)
          
    assert solution.theta.all() == 0 and solution.omega.all() == 0
    assert solution.x.all == 0 and solution.y.all() == - solution.pendulum.L

"""
test_solve_pendulum_ode_with_zero_ic()  








