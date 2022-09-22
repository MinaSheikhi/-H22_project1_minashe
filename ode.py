import numpy as np
import matplotlib.pyplot as plt

import abc
from fileinput import filename

from typing import NamedTuple, Optional, List
from scipy.integrate import solve_ivp



class InvalidInitialConditionError(RuntimeError):
    """"
    Make an Invalid Initial Condition Error.
    """
    pass

class ODEModel:
    @abc.abstractmethod  #gjør slik at det ikke går an å lage instans av metoden
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def num_states(self) -> int:
        raise NotImplementedError  

class ODEResult(NamedTuple): # Named tuple creates a immutable datastructure (once created cannot change)
        time: np.ndarray
        solution: np.ndarray

        @property
        def num_states(self) -> float:
            return self.solution.shape[0]

        @property
        def num_timepoints(self) -> float:
            return self.solution.shape[1]


def solve_ode(
    model: ODEModel,
    u0: np.ndarray,
    T: float,
    dt : float,
) -> ODEResult:

    if len(u0) != model.num_states:
        """
        Checks if u0 has as meny stats as conditioned in the model. 
        If not it returns Invalid Initial Condition Error.
        Solves u with the use of solve_ivp from scipy.integrate

        Argument
        --------
        uo : array
            Initialcondition array.

        Returns
        --------
        ODEResult(NamedTuple) containing timepoints and solution of ODE
        
        """
        raise InvalidInitialConditionError()

    t = np.arange(0, T, dt)
    sol = solve_ivp(fun=model, t_eval = t , t_span = (0, T), y0 = u0, method = "Radau")
    result = ODEResult(time = sol["t"], solution=sol["y"])
    
    return result
    

def plot_ode_solution(
    results: ODEResult,
    state_labels: Optional[List[str]] = None,
    filename: Optional[str] = None
) -> None:
    """
    Plots the ODE solution with timepoints.

    Arguments
    ---------
    results : ODEResult
            takes in results of the ODE and time

    state_labels : (Optional) List[str]
            This is optional. 
            Will be name of the states on the plot
        
    filename : (Optinal) str
            This is optional.
            File will be saved with this name.

    Output
    ---------
    Either saves file as filename or displays plot on the screen.
    """
    fig, ax = plt.subplots()
    sol = results.solution
    t = results.time

    if state_labels == None:
        state_labels = [("State " + str(i)) for i in range(results.num_states)]

    for i in range(len(sol)):
        if t.shape == sol[i].shape:
            ax.plot(t, sol[i], label=state_labels[i])

    ax.set_xlabel("time")
    ax.set_ylabel("ODE_solution")
    ax.grid(True)
    ax.legend()


    if filename is None:
        plt.show()
    else:
       fig.savefig(fname = filename) 