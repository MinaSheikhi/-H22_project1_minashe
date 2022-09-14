import numpy as np
from scipy.integrate import solve_ivp
from ode import ODEModel

class ExponentialDecay(ODEModel):
    def __init__(self, a: float):
        if (a < 0):
            raise ValueError ("a should be a positive integer")
        self.a = a

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        f = lambda t, u: -self.a * u
        return f(t,u)

    @property     # built in spesial function that creates and returns a property object.
    def decay(self): # Turning a into a property
        return self._a

    @decay.setter  #this makes the a variable private (local variable) with value x
    def decay(self, x):
        if (x < 0):
            raise ValueError ("Value is not positive.")
        self._a = x

    def num_states(self) -> int:
        return 1

def solve_exponential_decay():
    t = 0.3
    model = ExponentialDecay(0.4)
    u = np.array([3.2])
    du_dt = model.__call__(t,u)
    # array with routines based on numerical ranges
    sol = solve_ivp(du_dt, t_eval = np.arange(0, 10, 0.1) , t_span = (0,10) , y0 = (0,))
    return sol







        
        








