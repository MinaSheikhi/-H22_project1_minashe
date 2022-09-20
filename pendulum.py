import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from dataclasses import dataclass

from ode import ODEModel, solve_ode, plot_ode_solution, ODEResult


class Pendulum(ODEModel):
    def __init__(self, M: float = 1 , L: float = 1, g: float = 9.81) -> None:  #init is a constructor
        self.M = M
        self.L = L
        self.g = g

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:

        theta, omega = u
        dOmega_dt = (-self.g/self.L) * np.sin(theta)
        dtheta_dt = omega
         
        return np.array([dtheta_dt, dOmega_dt])

    @property   
    def num_states(self) -> int:
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
    def potential_energy(self) -> np.ndarray:  #hva slags t? self.result.time?
        return self.pendulum.g * (self.y + self.pendulum.L)
    
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
    u0: np.ndarray, 
    T: float, dt: float = 0.01, 
    pendulum: Pendulum = Pendulum()
 ) -> PendulumResults:

    result = solve_ode(u0 = u0, T = T, dt = dt, model = pendulum)
    pendulum_results = PendulumResults(result, pendulum)

    return pendulum_results

def plot_energy(results: PendulumResults, filename: Optional[str] = None) -> None:

    plt.plot(results.potential_energy, label = "Potential Energy")
    plt.plot(results.kinetic_energy, label = "Kinetic energy")
    plt.plot(results.total_energy, label = "Total energy")

    plt.title("ODE Results against time intervall")
    plt.ylabel("ODE solution")
    plt.legend()
    plt.grid(True)

    if filename != None:
        plt.savefig(fname = filename)
    else:
        plt.show()

    

def exercise_2g() -> PendulumResults:
    u0 = np.array([np.pi/6, 0.35])
    T = 10.0
    dt = 0.01
    
    sol = solve_pendulum(u0 = u0, T = T, dt = dt)
    return sol



"""
 Inherence can be very useful when creating specialized version of a class when 
 we only want to change on part of the class while keeping all the remaining funcitonality.
 """
class DampenedPendulum(Pendulum):
    def __init__(self, B, M: float = 1 , L: float = 1, g: float = 9.81) -> None:
        super().__init__(M, L, g)
        self.B = B

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        theta, omega = u
        dOmega_dt = (-self.g/self.L) * np.sin(theta) - self.B * omega
        dtheta_dt = omega
         
        return np.array([dtheta_dt, dOmega_dt])
    
def exercise_2h():
    u0 = np.array([np.pi/6, 0.35])
    T = 10.0
    dt = 0.01
    sol = solve_pendulum(u0 = u0, T = T, dt = dt)

    return sol



    



if __name__ == "__main__":
    #Exercise 2b
    sol_2b = exercise_2b(np.array([np.pi/6, 0.35]), 10.0, 0.01)
    plot_ode_solution(results = sol_2b, state_labels = [r"$\theta$", r"$\omega$"], filename = "exercise_2b.png")
   
"""
    #Exercise 2g
    sol_2g = exercise_2g()
    plot_energy(sol_2g, filename = "energy_single.png")
    "Yes the energy is conserved. This means that the total energy is always constant."

    #Exercise 2h
    sol_2h = exercise_2h()
    plot_energy(sol_2h, filename = "energy_damped.png")

"""










