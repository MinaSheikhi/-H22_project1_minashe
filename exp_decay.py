import numpy as np

class ExponentialDecay:
    def __init__(self, a: float):
        if (a < 0):
            raise ValueError ("a should be a positive integer")
        else:
            self.a = a

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        return - self.a * u




