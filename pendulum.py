import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from ode import ODEModel, solve_ode, plot_ode_solution, ODEResult


class Pendulum(ODEModel):
    def __init__(self, M: float = 1 , L: float = 1, g: float = 9.81):  #init is a constructor
        self.M = M
        self.L = L
        self.g = g

    def __call__(self, theta: float, omega: float):
        dOmega_dt = (-self.g/self.L) * np.sin(theta)
        dtheta_dt = omega
         
        return np.array([dtheta_dt, dOmega_dt])

    @property   
    def num_states(self):
        return 2


def exercise_2b(
    u0: np.ndarray,
    T: float,
    dt:float,
) -> ODEResult:
    model = Pendulum()
    result = solve_ode(model, u0, T, dt)
    return result




@dataclass #makes the class store data
class PendulumResults:
    results: ODEResult
    pendulum: Pendulum


    @property #this extracts the correct arrays from ode results
    def theta(self) -> np.ndarray:
        return self.results.solution[0]
        

    @property
    def omega(self) -> np.ndarray:
        return self.results.solution[1]
        
    @property
    def x(self) -> np.ndarray:
        return self.pendulum.L * np.sin(self.theta)

    @property
    def y(self) -> np.ndarray:
        return - self.pendulum.L * np.cos(self.theta)

    @property
    def potential_energy(self, t) -> np.ndarray:  #hva slags t? self.result.time?
        P = lambda t: self.pendulum.g * (self.y + self.pendulum.L)
        return P
    
    @property
    def vx(self) -> np.ndarray:
        return np.gradient(self.x, self.results.time)

    @property
    def vy(self) -> np.ndarray:
        return np.gradient(self.y, self.results.time)

    @property
    def kinetic_energy(self) -> np.ndarray:
        return (1/2 * (self.vx**2 + self.vy**2))

    @property
    def total_energy(self):
        return self.potential_energy + self.kinetic_energy


def solve_pendulum(
    u0: np.ndarray, T: float, dt: float = 0.01, pendulum: Pendulum = Pendulum()
    ) -> PendulumResults:
    return solve_ode(u0 = u0, T = T, dt = dt, model = pendulum)


def plot_energy(results: PendulumResults, filename: str = None) -> None:
    plt.plot(results.time, results.potential_energy, label = "Potential Energy")
    plt.plot(results.time, results.kinetic_energy, label = "Kinetic energy")


    plt.legend()

    plt.grid(True)

    plt.show()

    plt.savefig(filename)

"""
def exercise_2g():
    u0 = np.array(np.pi, 0.35)
    T = 10.0
    dt = 0.01
    
    sol = solve_pendulum(u0 = u0, T = T, dt = dt)
    return plt.energy(sol, filename = "energy_single.png")


"Yes the energy is conserved. This means that the total energy is always constant."


class DampenedPendulum(Pendulum):
    pass

if __name__=="__main__":
    #Exercise 2b
    solution = exercise_2b([np.pi/6, 0.35], 10.0, 0.01)
    print(solution)
    plot_ode_solution(results = result, state_labels = ["theta", "omega"], filename = "exercise_2b.png")
    #Exercise 2g
    exercise_2g()
"""

solution = exercise_2b(u0 = np.array([np.pi/6, 0.35]), T= 10.0, dt = 0.01)
print(np.array(solution))











