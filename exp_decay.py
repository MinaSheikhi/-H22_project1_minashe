import numpy as np
from scipy.integrate import solve_ivp
from ode import ODEModel, solve_ode, plot_ode_solution

class ExponentialDecay(ODEModel):
    def __init__(self, a: float):
        """
        Constructor for ExponentialDecay
        
        Input
            -a: float,
            the decay constant, this number cannot be negative.

        Raises: ValueError: if input value is negative
        
        """
        if (a < 0):
            raise ValueError ("Value is not positive.")
        self.a = a

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:

        ''' 
        Call function that returns derivative of u at time t.
        
        Input
            - t: float
            - u: numpy array

        Output
            - a numpy array, the derivative dudt '''
        

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

    @property
    def num_states(self) -> int:
        """
        Returns number of state variables in ODE.

        Output
            - an integer
        """
        return 1

def solve_exponential_decay(
    a: float = .4, 
    u: np.ndarray = np.array([3.2]), 
    u0: tuple = (0, ), 
    T: int = 10, 
    dt: float = .1, 
) -> np.ndarray:

    """
    Instantiates the ExponentialDecay class and solves the ode using solve_ivp from scipy library.
    Default solves for 
        f(t = 0.4, u = [3.2]) with u0 = (0, ), T = 10, dt = 0.1
    Input
        - a: float
            the decay constant
        - u: np.ndarray
        - u0: tuple
            initial condition for u
        - T: int
            end time
        - dt: float
            timestep
            
    Output
        - an array
            Solved ode
    """

    model = ExponentialDecay(a)
    t = np.arange(0, T, dt)
    sol = solve_ivp(fun=model, t_span=(0,T), y0=u0, t_eval=t)
    return sol.t, sol.y

model = ExponentialDecay(0.4)
result = solve_ode(model, u0=np.array([4.0]), T=10.0, dt=0.01)
plot_ode_solution(
    results=result, state_labels=["u"], filename="exponential_decay.png")


if __name__=="__main__":
     solve_exponential_decay(0.4, np.array([3.2]), (0, ), 10, 0.01) # Exercise 1e



        
        








