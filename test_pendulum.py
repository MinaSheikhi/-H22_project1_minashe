import numpy as np
from pendulum import Pendulum, exercise_2b


"""@@pytest.mark.parametrize("L, gamma, theta, solution", [1.42, 0.35, np.pi/6, ], [1.42, 0, 0, 0])
def test_Pendulum(L, gamma, theta):
    model = Pendulum()
    dTheta = model(theta, gamma)
    solution =
    assert """

def test_solve_pendulum_ode_with_zero_ic():
    res = exercise_2b(u0 = np.array([0, 0]))

    assert res.result.all() == 0




    








