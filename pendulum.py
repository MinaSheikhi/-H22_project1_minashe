import numpy as np
from ode import ODEModel, solve_ode, plot_ode_solution


class Pendulum(ODEModel):
    def __init__(self, M: float = 1 , L: float = 1, g: float = 9.81):  #init is a constructor
        self.M = M
        self.L = L
        self.g = g

    def __call__(self, theta: float, omega: float):
        dOmega_dt = lambda t: - self.g * np.sin(theta) / self.L
        dtheta_dt = lambda t: omega
         
        return dtheta_dt

    @property   
    def num_states(self):
        return 2


def exercise_2b(
    u0: np.ndarray = [np.pi/6, 0.35],
    T: float = 10.0,
    dt:float = 0.01,
):
    model = Pendulum()
    result = solve_ode(model, u0, T, dt)

    plot_ode_solution(results = result, state_labels = ["theta", "omega"], filename = "exercise_2b.png")


if __name__ == "__main__":
    exercise_2b([np.pi/6, 0.35], 10.0, 0.01)






