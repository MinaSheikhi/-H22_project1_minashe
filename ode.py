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
    if len(u0) == model.num_states:
        t = np.arange(0, T, dt)
        sol = solve_ivp(fun=model, t_eval = t , t_span = (0,T) , y0 = u0)
        result = ODEResult(time = sol["t"], solution = sol["y"])
        return result
    else:
        raise InvalidInitialConditionError()


def plot_ode_solution(
    results: ODEResult,
    state_labels: Optional[List[str]] = None,
    filename: Optional[str] = None
) -> None:
   
    plt.xlabel("time")
    plt.ylabel("ODE_solution")

    plt.plot(results, state_labels)


    plt.grid()

    plt.legend()

    plt.show()

    plt.savefig('filename')


   




