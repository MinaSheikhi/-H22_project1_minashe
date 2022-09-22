import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from dataclasses import dataclass

from ode import ODEModel, ODEResult, solve_ode, plot_ode_solution


class Pendulum(ODEModel):
    def __init__(self, M: float = 1 , L: float = 1, g: float = 9.81) -> None:  #init is a constructor

        """
        Takes inn input arguments.
        
        Arguments
        ----------
        M : float
            Mass
        L : float
            Length
        g : float
            gravitation 

        Returns
        ----------
        Nothing

        """

        self.M = M
        self.L = L
        self.g = g

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:

        theta, omega = u
        dOmega_dt = (-self.g/self.L) * np.sin(theta)
        dtheta_dt = omega

        """
        Call function that returns derivative of u at time t.

        Arguments
        ---------
        t : float
            time

        u: array
            consisting of states theta and omega

        Returns
        ---------
        The derivatives of theta and omega

        """
         
        return np.array([dtheta_dt, dOmega_dt])

    @property   
    def num_states(self) -> int:
        """
        Returns output of states.

        Output:
            - integer
        """

        return 2


def exercise_2b(
    u0: np.ndarray,
    T: float,
    dt:float,
) -> ODEResult:

    """
    creating Pendulum instance and solving the ODEs.
    
    Arguments
    ----------
    u0 : array
        Start values for the initial states.

    T : float
        End value for the time

    dt : float
        Time step

    Returns
    ----------
    The solution of the ODEs.
    """
    u0 = np.array([np.pi/6, 0.35])
    T = 10.0
    dt = 0.01
    
    result = solve_ode(model = Pendulum(), u0 = u0, T = T, dt = dt)

    plot_ode_solution(results = result, state_labels = [r"$\theta$", r"$\omega$"], filename = "exercise_2b.png")
    return result


@dataclass # Makes the class store data
class PendulumResults:
    results: ODEResult
    pendulum: Pendulum

    @property 
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
        return (1/2*(self.vx**2 + self.vy**2))

    @property
    def total_energy(self) -> np.ndarray:
        return self.potential_energy + self.kinetic_energy


def solve_pendulum(
    u0: np.ndarray, 
    T: float, dt: float = 0.01, 
    pendulum: Optional[Pendulum] = None
 ) -> PendulumResults:

    """
    Solves the ODE of pendulum and stores result in an instance of the PendulumResults.

    Arguments
    ---------
    u0 : array
        Start values for the initial states.

    T : float
        End value for the time

    dt : float
        Time step

    pendulum : Pendulum
                The ODE model called pendulum

    Returns
    ---------
    The solution of the ODE and the time given chosen model and variables.

    """

    result = solve_ode(u0 = u0, T = T, dt = dt, model = pendulum)
    pendulum_results = PendulumResults(result, pendulum)

    return pendulum_results

solve_ode(u0 = np.array([np.pi/6, 0.35]), T = 10.0, dt = 0.01, model = Pendulum())

def plot_energy(results: PendulumResults, filename: Optional[str] = None) -> None:

    """
    Plots the potential, kinetic and total energy on y axis and time on x-axis.

    Arguments
    ---------
    reuslts : PendulumResults
    filename : str

    Output
    --------
    Either saves the file if it does not already exist. If it exists displays the plot on the screen.
    Here we use subplots method which creates the figure along iwth the subplots that are stored in 
    the ax array. fig is matplotlib.figure.Figure
    

    """
    fig, ax = plt.subplots()
    ax.plot(results.results.time, results.potential_energy, label = "Potential Energy")
    ax.plot(results.results.time, results.kinetic_energy, label = "Kinetic energy")
    ax.plot(results.results.time, results.total_energy, label = "Total energy")

    ax.set_title("ODE Results against time intervall")
    ax.set_ylabel("ODE solution")
    ax.set_xlabel("Time")
    ax.grid(True)
    ax.legend()
    
    if filename != None:
        fig.savefig(fname = filename)
    else:
        plt.figure()


def exercise_2g():

    """
    Solves the pendulum equation using the values given and plots the results.

    u0 : array
        Start values for the initial states.

    T : float
        End value for the time

    dt : float
        Time step
    
    Output
    --------
    Saves the plot of energy in a file energy_single.png.

    """

    u0 = np.array([np.pi/6, 0.35])
    T = 10.0
    dt = 0.01
    
    sol_2g = solve_pendulum(u0 = u0, T = T, dt = dt, pendulum = Pendulum())
    plot_energy(sol_2g, filename = "energy_single.png")



"""
 Inherence can be very useful when creating specialized version of a class when 
 we only want to change on part of the class while keeping all the remaining funcitonality.
"""
class DampenedPendulum(Pendulum):
    def __init__(self, B: float, M: float = 1 , L: float = 1, g: float = 9.81) -> None:

        """
        Takes in inputs from the superclass in addition to B

        Arguments
        ----------
        B : float
            Damping parameter

        Returns
        ---------
        Nothing
        """

        super().__init__(M, L, g)
        self.B = B

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:

        """
        Call function that returns the drivatives of u at time t.

        Arguments
        ----------
        t : float
            time
        u : array
            consisting of the states theta and omega.

        Returns
        ---------
        Array with the derivatives.
        """


        theta, omega = u
        dOmega_dt = (-self.g/self.L) * np.sin(theta) - self.B * omega
        dtheta_dt = omega
         
        return np.array([dtheta_dt, dOmega_dt])
    
def exercise_2h():
    """"
    Solves and plots the energy with given values. Saves the plot with the filename energy_damped.png

    Output:
        - Saved plot with given filename

    """

    u0 = np.array([np.pi/6, 0.35])
    T = 10.0
    dt = 0.01
    sol_2h = solve_pendulum(u0, T, dt, DampenedPendulum(1))
    plot_energy(sol_2h, filename = "energy_damped.png")
 

if __name__ == "__main__":
    #Exercise 2b
    exercise_2b(np.array([np.pi/6, 0.35]), 10.0, 0.01)
   
   
    #Exercise 2g
    exercise_2g()
    "Yes the energy is conserved. This means that the total energy is always constant."

    #Exercise 2h
    exercise_2h()
