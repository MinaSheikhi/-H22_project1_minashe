import numpy as np
import abc
from typing import NamedTuple
from scipy.integrate import solve_ivp

class ODEModel(abc.ABC):

    @abc.abstractmethod
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def num_states(self) -> int:
        raise NotImplementedError  #super-class method is not implemented and child classes should implement it

class ODEResult(NamedTuple): # Named tuple creates a immutable datastructure (once created cannot change)
    time: np.ndarray
    solution: np.ndarray

class InvalidInitialConditionError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)

def solve_ode(model: ODEModel, u0: np.ndarray, T: float, dt: float) -> ODEResult:
    if (len(u0) == model.num_states):
        sol = solve_ivp(model.du_dt, t_eval = np.arange(0, 10, 0.1) , t_span = (0,10) , y0 = (0,))
        result = ODEResult(time = sol["t"], solution = sol["y"])
        return result
    else:
        raise InvalidInitialConditionError(f"u0 needs to match the number of state variables of {type(model).__name__}. Len(u0) = {len(u0)}, and num_states = {model.num_states}.")


