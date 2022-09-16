from xml import dom
import numpy as np
from dataclasses import dataclass
from ode import ODEModel, ODEResult, solve_ode, plot_ode_solution
from exp_decay import ExponentialDecay
from typing import Optional
import matplotlib.pyplot as plt


class Pendulum(ODEModel):
    def __init__(self, L: float = 1, M: float = 1, g: float = 9.81):
        self.M = M #mass
        self.L = L #length
        self.g = g #gravitional acceleration

    def __call__(self, t, u):
        theta, omega = u
        dtheta = omega
        domega = (-self.g/self.L)*np.sin(theta)
        return np.array([dtheta, domega])
        
    @property
    def num_states(self):
        return 2


def exercise_2b(
    u0: np.ndarray,
    T: float,
    dt: float,
) -> ODEResult:

    model = Pendulum()
    result = solve_ode(model, u0, T, dt)
    return result


@dataclass
class PendulumResults:
    results: ODEResult
    pendulum: Pendulum

    @property
    def theta(self) -> np.ndarray:
        return self.results.solution ########????????###############

    @property 
    def omega(self) -> np.ndarray:
        return self.results.time ########?????????##############

    @property
    def x(self) -> np.ndarray:
        return self.pendulum.L*np.sin(self.theta)

    @property 
    def y(self) -> np.ndarray:
        return -self.pendulum.L*np.cos(self.theta)

    @property 
    def potential_energy(self) -> np.ndarray:
        return self.pendulum.g*(self.y + self.pendulum.L)

    @property
    def vx(self) -> np.ndarray:
        return np.gradient(self.x, self.results.time)

    @property 
    def vy(self) -> np.ndarray:
        return np.gradient(self.y, self.results.time) ##########????#####
    
    @property
    def kinetic_energy(self) -> np.ndarray:
        return (1/2)*(self.vx**2 + self.vy**2)

    @property
    def total_energy(self) -> np.ndarray:
        return self.potential_energy + self.kinetic_energy
    

def solve_pendulum(
    u0: np.ndarray,
    T: float,
    dt: float,
    pendulum: Optional[Pendulum] = Pendulum() ## et sted sier oppgave default skal være None, et annet sted står det at det skal være Pendulum()?
) -> PendulumResults:

    result = solve_ode(pendulum, u0, T, dt)
    pendulum_results = PendulumResults(result, pendulum)
    return pendulum_results

def plot_energy(results: PendulumResults, filename: Optional[str] = None) -> None:
    pass

def exercise_2g() -> None:
    u0 = np.array([np.pi/6, .35])
    T = 10
    dt = .01

    result = solve_pendulum(u0, T, dt)
    plot_energy(result, "energy_single.png")
    """
    The energy is conserved.
    """


class DampenedPendulum(Pendulum):
    def __init__(self, B):
        self.B = B

    def __call__(self, theta, omega):
        return -self.g/self.L*np.sin(theta) - self.B*omega


def exercise_2h():
    u0 = np.array([np.pi/6, .35])
    T = 10
    dt = .01

    result = solve_pendulum(u0, T, dt, DampenedPendulum(1))
    plot_energy(result, "energy_damped.png")
    """
    ############################TODO############################
    """

if __name__ == "__main__":
    res_2b = exercise_2b(np.array([np.pi/6, .35]), 10, .01)

    state_labels = [r"$\theta$", r"$\omega$"]
    plot_ode_solution(res_2b, state_labels, "exercise_2b.png")

    exercise_2g()
    exercise_2h()



    
