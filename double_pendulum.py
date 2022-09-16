from ode import ODEModel
import numpy as np

class DoublePendulum(ODEModel):
    def __init__(self, L1: float, L2: float) -> None:
        self.L1 = L1
        self.L2 = L2
    
    def __call__(self, t: float, u: np.ndarray):
        theta1, omega1, theta2, omega2 = u
        dtheta1_dt = omega1
        dtheta2_dt = omega2
        dtheta = dtheta2_dt - dtheta1_dt
        domega1_dt = (self.L1*omega1**2*np.sin(dtheta)*np.cos(dtheta) + self.g*np.sin(theta2)*np.cos(dtheta) + self.L2*omega2**2*np.sin(dtheta) - 2*self.g*np.sin(theta1))/(2*self.L1 - self.L1*np.cos**2(dtheta))
        domega2_dt = (-self.L2*omega2**2*np.sin(dtheta)*np.cos(dtheta) + 2*self.g*np.sin(theta1)*np.cos(dtheta) - 2*self.L1*omega1**2*np.sin(dtheta) - 2*self.g*np.sin(theta2))/(2*self.L2 - self.L2*np.cos**2(dtheta))
        
        return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])

    @property 
    def num_states(self) -> int:
        return 4