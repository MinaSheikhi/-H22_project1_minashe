import numpy as np
import abc
from typing import NamedTuple
from scipy.integrate import solve_ivp
from exception import InvalidInitialConditionError

class ODEModel(abc.ABC):

    @abc.abstractmethod
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def num_states(self) -> int:
        raise NotImplementedError

class ODEResult(NamedTuple):
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
        raise InvalidInitialConditionError(f"u0 needs to match the number of state variables of {type(model).__name__}. Len(u0) = {len(u0)}, and num_states = {model.num_states}.")

    t = np.arange(0, T, dt)
    sol = solve_ivp(fun=model, t_span=(0,T), y0=u0, t_eval=t)
    res = ODEResult(time=sol["t"], solution=sol["y"])

    return res

