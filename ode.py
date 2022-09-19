import abc
from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple, Optional, List
from scipy.integrate import solve_ivp


class InvalidInitialConditionError(RuntimeError):
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
        def num_states(self):
            return self.solution.shape[0]

        @property
        def num_timepoints(self):
            return self.time.shape[0]


def solve_ode(
    model: ODEModel,
    u0: np.ndarray,
    T: float,
    dt : float,
) -> ODEResult:

    if len(u0) != model.num_states:
        raise InvalidInitialConditionError()

    t = np.arange(0, T, dt)
    sol = solve_ivp(fun=model, t_eval = t , t_span = (0, T), y0 = u0, method = "Radau")
    result = ODEResult(time = t, solution = sol.y)
    
    return result
    

def plot_ode_solution(
    results: ODEResult,
    state_labels: Optional[List[str]] = None,
    filename: Optional[str] = None
) -> None:
   
    sol = results.solution
    t = results.time

    if state_labels == None:
        state_labels = [("State " + str(i)) for i in range(results.num_states)]

    for i in range(len(sol)):
        if t.shape == sol[i].shape:
            plt.plot(t, sol[i], label=state_labels[i])

    plt.xlabel("time")
    plt.ylabel("ODE_solution")
    plt.grid(True)
    plt.legend()


    if filename != None:
        plt.savefig(fname = filename)
    else:
        plt.show()

   




