import numpy as np
import abc

class ODEModel(abc.ABC):
    
    @abc.abstractmethod
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError