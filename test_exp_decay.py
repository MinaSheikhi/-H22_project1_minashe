import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import pytest

from exp_decay import ExponentialDecay
from ode import ODEModel, solve_ode, InvalidInitialConditionError, ODEResult, plot_ode_solution



model = ExponentialDecay(0.4)

def test_ExponentialDecay():
    t = 0.0
    u = np.array([3.2])
    du_dt = model(t,u)
    
    assert math.isclose(du_dt[0],-1.28)


def test_negative_decay_raises_ValueError_1a():
    with pytest.raises(ValueError):
        model = ExponentialDecay(-0.4)

def test_negative_decay_raises_ValueError_1b():
    with pytest.raises(ValueError):
        model.decay = -1.0  # cant set a to new value -1 because of valueerror
              
def test_num_states():
    assert model.num_states == 1


def test_solve_with_different_number_of_initial_states():
    with pytest.raises(InvalidInitialConditionError):
        solve_ode(model, u0 = np.array([0,1]), T = 10.0, dt = 0.2)

@pytest.mark.parametrize("a, u0, T, dt", [
    (3, np.array([10]), 1, 0.1),
    (4, np.array([3]), 10, 0.1),
    (0, np.array([6]), 12, 0.2)])

def test_solve_time(a, u0, T, dt):
    model = ExponentialDecay(a)
    res = solve_ode(model, u0, T, dt) 

    t_computed = res.time  #res.time gives the timepoints the ODE is correct
    t_expected = np.arange(0, T, dt) # Evalueate timepoints manually
    
    assert t_expected.all() == t_computed.all() # .all() goes though each of the numbers in the array and (==) compares

@pytest.mark.parametrize("a, u0, T, dt", [
    (3, np.array([10]), 1, 0.1),
    (4, np.array([3]), 10, 0.1),
    (0, np.array([6]), 12, 0.2)])

def test_solve_solution(a, u0, T, dt):

    model = ExponentialDecay(a)
    res = solve_ode(model, u0, T, dt)
    t = np.arange(0, T, dt)

    y = res.solution
    y_exact = u0 * np.exp(-a*t)
    relative_error = np.linalg.norm(np.subtract(y, y_exact)) / np.linalg.norm(y_exact)

    assert (relative_error < 0.01)

def test_ODEResults():
    results = ODEResult(time=np.array([0, 1, 2]), solution=np.zeros((2, 3)))

    assert (results.num_states == 2 and results.num_timepoints == 3)



def test_function_that_creates_a_file():

    # Check if the file already exists and delete if necessary
    filename = Path("test_plot.png")
    if filename.is_file():
        filename.unlink()

    # Call the function we are testing
   plt.savefig("filename")

    # Check that the file has now been created, then delete it
    assert filename.is_file()
    filename.unlink()

def test_plot_ode_solution_saves_file():

    model = ExponentialDecay(0.4)
    result = solve_ode(model, u0=np.array([4.0]), T=10.0, dt=0.01)
    plot_ode_solution(
    results=result, state_labels=["u"], filename="exponential_decay.png")
    
    filename = Path("exponential_decay.png")
    # Check if the file already exists and delete if necessary
    if filename.is_file():
        filename.unlink()

    # Check that the file has now been created, then delete it
    assert filename.is_file()
    filename.unlink()
