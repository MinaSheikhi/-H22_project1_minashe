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

def solve_ode(
    model: "ODEModel",
    u0: np.ndarray,
    T: float,
    dt : float, 
) -> "ODEResult":

    t = np.arange(0, T, dt)
    f = model(t, u0)
    sol = solve_ivp(fun=f, t_span=(0,T), y0=u0, t_eval=t)
    res = ODEResult(time=sol["t"], solution=sol["y"])

    return res

class ODEResult(NamedTuple):
    time: np.ndarray
    solution: np.ndarray