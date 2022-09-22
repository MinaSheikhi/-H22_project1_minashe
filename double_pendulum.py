from fileinput import filename
import numpy as np
from typing import Optional
from dataclasses import dataclass

from ode import ODEModel, ODEResult, solve_ode
from pendulum import plot_energy


class DoublePendulum(ODEModel):
    def __init__(self, L1: float = 1, L2: float = 1, M1: float = 1, M2: float = 1, g: float = 9.81) -> float:
        self.L1 = L1
        self.L2 = L2
        self.M1 = M1
        self.M2 = M2
        self.g = g

    def __call__(self, t:float, u:np.ndarray) -> np.ndarray:
        theta1, omega1, theta2, omega2 = u

        dtheta1_dt = omega1
        dtheta2_dt = omega2
        dtheta_dt = theta2 - theta1
        
        domega1_dt = ((self.L1 * (omega1**2) * np.sin(dtheta_dt) * np.cos(dtheta_dt) + self.g * np.sin(theta2) * np.cos(dtheta_dt) + self.L2 * (omega2**2) * np.sin(dtheta_dt) - 2 * self.g * np.sin(theta1)) / (2 * self.L1 - self.L1 * (np.cos(dtheta_dt)**2)))
        domega2_dt = ((-self.L2 * (omega2**2) * np.sin(dtheta_dt) * np.cos(dtheta_dt) + 2 * self.g * np.sin(theta1) * np.cos(dtheta_dt) - 2 * self.L1 * (omega1**2) * np.sin(dtheta_dt) - 2 * self.g * np.sin(theta2)) / (2 * self.L2 - self.L2 * (np.cos(dtheta_dt)**2)))

        return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])

    @property
    def num_states(self) -> int:
        return 4


@dataclass #makes the class store data
class DoublePendulumResults:
    results: ODEResult
    pendulum: DoublePendulum

    @property
    def theta1(self) -> np.ndarray:
        return self.results.solution[0]

    @property
    def omega1(self) -> np.ndarray:
        return self.results.solution[1]

    @property
    def theta2(self) -> np.ndarray:
        return self.results.solution[2]

    @property
    def omega2(self) -> np.ndarray:
        return self.results.solution[3]

    @property
    def x1(self) -> np.ndarray:
        return self.pendulum.L1 * np.sin(self.theta1)

    @property
    def y1(self) -> np.ndarray:
        return - self.pendulum.L1 * np.cos(self.theta1)

    @property
    def x2(self) -> np.ndarray:
        return self.x1 + self.pendulum.L2 * np.sin(self.theta2)

    @property
    def y2(self) -> np.ndarray:
        return self.y1 - self.pendulum.L2 * np.cos(self.theta2)

    @property
    def potential_energy(self) -> np.ndarray:
        P1 = self.pendulum.M1*self.pendulum.g*(self.y1 + self.pendulum.L1)
        P2 = self.pendulum.M2*self.pendulum.g*(self.y2 + self.pendulum.L1 + self.pendulum.L2)
        P = P1 + P2
        return P

    @property
    def vx1(self) -> np.ndarray:
        return np.gradient(self.x1, self.results.time)

    @property
    def vy1(self) -> np.ndarray:
        return np.gradient(self.y1, self.results.time)
    @property
    def vx2(self) -> np.ndarray:
        return np.gradient(self.x2, self.results.time)

    @property
    def vy2(self) -> np.ndarray:
        return np.gradient(self.y2, self.results.time)

    @property
    def kinetic_energy(self) -> np.ndarray:
        K1 = 0.5* self.pendulum.M1*((self.vx1**2) + (self.vy1**2))
        K2 = 0.5 * self.pendulum.M2*((self.vx2**2) + (self.vy2**2))
        K = K1 + K2
        return K

    @property
    def total_energy(self) -> np.ndarray:
        return self.potential_energy + self.kinetic_energy


def solve_double_pendulum(
    u0: np.ndarray,
    T: float,
    dt: float = 0.01,
    pendulum: Optional[DoublePendulum] = None,
) -> DoublePendulumResults:


    result = solve_ode(u0 = u0, T = T, dt = dt, model = pendulum)
    pendulum_results = DoublePendulumResults(result, pendulum)

    return pendulum_results

def exercise_3d() -> np.ndarray:
    u0 = np.array([np.pi/6, 0.35, 0, 0])
    T = 10.0
    dt = 0.01
    results = solve_double_pendulum(u0 = u0, T = T, dt = dt, pendulum = DoublePendulum())

    plot_energy(results, filename = "energy_double.png")

if __name__ == "__main__":
    exercise_3d()
 