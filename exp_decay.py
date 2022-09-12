import numpy as np
from ode import ODEModel

class ExponentialDecay(ODEModel):
    def __init__(self, a):
        self.decay = a
    
    @property
    def decay(self):
        return self._a

    @decay.setter
    def decay(self, value):
        if value < 0:
            raise ValueError("Constant cannot be negative.")
        self._a = value
    
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        f = lambda t, u: -self.decay*u
        return f(t, u)
