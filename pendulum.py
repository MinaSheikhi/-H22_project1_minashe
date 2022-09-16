import numpy as np
from dataclasses import dataclass
from ode import ODEModel, ODEResult, solve_ode, plot_ode_solution
from exp_decay import ExponentialDecay


class Pendulum(ODEModel):
    def __init__(self, L: float = 1, M: float = 1, g: float = 9.81):
        self.M = M #mass
        self.L = L #length
        self.g = g #gravitional acceleration

    #def __call__(self, u):
    def __call__(self, theta, omega):
        f = lambda theta, omega: (-self.g/self.L)*np.sin(theta)
        return f(theta, omega)
        
    @property
    def num_states(self):
        return 2


def exercise_2b(
    L: float,
    u0: np.ndarray,
    T: float,
    dt: float,
) -> ODEResult:

    model = Pendulum(L)
    result = solve_ode(model, u0, T, dt)
    return result

@dataclass
class PendulumResults:
    results: ODEResult
    pendulum: Pendulum

    @property
    def theta(self):
        return self.results.solution ########????????###############

    @property 
    def omega(self):
        return self.results.time ########?????????##############

#     @property
#     def x(self) -> np.ndarray:
#         return self.pendulum.L*np.sin(self.theta)

#     @property 
#     def y(self) -> np.ndarray():
#         return -self.pendulum.L*np.cos(self.theta)


# def solve_pendulum(
#     u0: np.ndarray,
#     T: float,
#     dt: float,
#     pendulum: Optional[Pendulum] = Pendulum() ## et sted sier oppgave default skal være None, et annet sted står det at det skal være Pendulum()?
# ) -> PendulumResults:

#     pass

if __name__ == "__main__":
    res = exercise_2b(1.42, np.array([np.pi, .35]), 10, .01)

    state_labels = [r"$\theta$", r"$\omega$"]
    plot_ode_solution(res, state_labels, "exercise_2b.png")