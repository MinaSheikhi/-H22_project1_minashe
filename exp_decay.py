import numpy as np
import cmath
from ode import ODEModel
from scipy.integrate import solve_ivp


class ExponentialDecay(ODEModel):
    def __init__(self, a) -> None:
        """
        Constructor for ExponentianlDecay. 

        Input
            - a: float,
            the decay constant, this number cannot be negative.
        
        Raises:
            ValueError: if input value is negative.
        """
        self.decay = a
    
    @property
    def decay(self) -> float:
        return self._a

    @decay.setter
    def decay(self, value) -> None:
        if value < 0:
            raise ValueError("Constant cannot be negative.")
        self._a = value
    
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """
        Call function that returns the derivative of u at time t. 

        Input
            - t: float
            - u: numpy array

        Output
            - a numpy array, the derivate dudt
        """

        f = lambda t, u: -self.decay*u
        return f(t, u)

    @property
    def num_states(self) -> int:
        """
        Returns number of state variables in ODE.

        Output
            - an int
        """

        return 1
    
def solve_exponentional_decay(
    a: float = .4, 
    u: np.ndarray = np.array([3.2]), 
    u0: tuple = (0, ), 
    T: int = 10, 
    dt: float = .1, 
) -> np.ndarray:

    """
    Instantiates the ExponentialDecay class and solves the ode using solve_ivp from scipy library.
    Solves for 
        f(t = 0.4, u = [3.2])

    Output
        - an array
            Solved ode
    """

    modell = ExponentialDecay(a)
    t = np.arange(0, T, dt)
    sol = solve_ivp(fun=modell, t_span=(0,T), y0=u0, t_eval=t)
    return sol["t"], sol["y"]