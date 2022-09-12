import numpy as np

class ExponentialDecay:
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
        return -self._a*u

