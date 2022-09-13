import numpy as np

# Oppgave 1a

class ExponentialDecay:
    def __init__(self, a: float):
        if (a < 0):
            raise ValueError ("a should be a positive integer")
        self.a = a

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        return - self.a * u
       

    @property     # built in spesial function that creates and returns a property object.
    def decay(self): # Turning a into a property
        return self._a

    @decay.setter  #this makes the a variable private (local variable) with value x
    def decay(self, x):
        if (x < 0):
            raise ValueError ("Value is not positive.")
        self._a = x




