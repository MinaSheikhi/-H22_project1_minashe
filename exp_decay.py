import numpy as np

class ExponentialDecay:
    def __init__(self, a):
        if abs(a) != a:
            raise ValueError("Constant cannot be negative.")
        self.a = a

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        return -self.a*u
