""" 
written by: Jonathan Pilgram, november '23 - august '24
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

"""Collection of functions to create a reduced order basis"""


def spatial_snapshots(decay_chain, N_snapshots: int) -> None:
    """Creates evenly spaced spatial snapshots of the Full Order Model.
    The columns represent a spatial state of the system at a certain moment.

    Parameters
    ----------
    N_snapshots : int
        The amount of snapshots to generate

    data saved
    ----------
    snapshots : np.array
        Array with the snapshots of the simulation
    """

    snap_times = np.linspace(
        0, len(decay_chain.solution.y[0, :]) - 1, N_snapshots, dtype=int
    )
    if decay_chain.full_info == True:
        print("Snapshots are taken on:")
        print(snap_times)
    decay_chain.snapshots = decay_chain.solution.y[:, snap_times]


def reduce_SVD(decay_chain, order: int = 0) -> None:
    """Makes a POD using an SVD

    parameters
    ----------
    order (default 0) : int
        Controls the order of the reduced order model. 0 -> full order

    data saved
    ----------
    decay_array_red : np.array of size order x order
        The reduced order decay array
    initial_composition_red : np.array of size order
        Reduced space initial composition
    N_nuclides_red : int
        Amount of quasi-nuclides used for the reduced order model

    methods added
    -------------
    run_simulation_reduced
        Similar to run_simulation for running the reduced order model
        # Do not use, just call rom.run_simulation_reduced
    _expand
        The method to transform back from the reduced order space to full space
    """

    pod_basis, decay_chain.singular_values, right_side = np.linalg.svd(
        decay_chain.snapshots, full_matrices=False
    )
    if decay_chain.full_info == True:
        print("Shape left side:" + str(np.shape(pod_basis)))
        print("Shape singular values:" + str(np.shape(decay_chain.singular_values)))
        print("Shape right side:" + str(np.shape(right_side)))

    if order == 0:
        order = len(decay_chain.singular_values)
    decay_chain.rom_basis = pod_basis[:, 0:order]

    if decay_chain.info == True:
        plt.title("Singular values")
        plt.semilogy(decay_chain.singular_values)
        plt.show()
        plt.xlabel("Time (s)")
        plt.ylabel("Nuclide amount")
        print("Shape pod_basis:" + str(np.shape(decay_chain.rom_basis)))
    decay_chain.decay_array_red = (
        decay_chain.rom_basis.T @ decay_chain.decay_array @ decay_chain.rom_basis
    )
    decay_chain.initial_composition_red = (
        decay_chain.rom_basis.T @ decay_chain.initial_composition
    )
    decay_chain.N_nuclides_red = order
    decay_chain.run_simulation_reduced = run_simulation_reduced
    decay_chain._expand = lambda y: decay_chain.rom_basis @ y


def reduce_abstract(decay_chain, order: int) -> None:
    """Creates all the objects needed to do a reduction without a reduction method

    parameters
    ----------
    order : int
        Controls the order of the reduced order model.

    methods added
    -------------
    _expand
        The method to transform back from the reduced order space to full space
    """
    decay_chain.N_nuclides_red = order
    decay_chain._expand = lambda y: decay_chain.rom_basis @ y


def run_simulation_reduced(decay_chain, log_scale: bool = False):
    """Function to run the reduced order simulation

    data added
    ----------
    solution_red : data object
        solution_red.t : the timesteps at which the solution is evaluated
        solution_red.y : the solution values
    simulation_time_rom : float
        The time needed for the reduced order simulation in seconds
    log_scale : bool (default False)
        Whether to plot the yscale logarithmitically. Useful for MSFR burnup.
    methods added
    -------------
    _plot_reduced_space
        Private method to plot the solution in the reduced order space
    _plot_reduced_solution
        Private method to plot the reduced order solution"""

    if np.all(decay_chain.initial_composition == 0):
        print("Did you forget to set initial_composition?")
    decay_chain._plot_reduced_space = plot_reduced_space
    decay_chain._plot_reduced_solution = plot_reduced_solution

    start = time.time()

    if decay_chain.solver == "Euler_forward":
        solution = decay_chain._forward_euler(
            decay_chain.decay_array_red, decay_chain.initial_composition_red
        )
    elif decay_chain.solver == "Euler_backward":
        solution = decay_chain._backward_euler(
            decay_chain.decay_array_red, decay_chain.initial_composition_red
        )
    else:
        solution = decay_chain._run_simulation(
            decay_chain.decay_array_red,
            decay_chain.initial_composition_red,
            decay_chain.solver,
        )

    decay_chain.simulation_time_rom = time.time() - start
    if decay_chain.info == True:
        print("Simulation time ROM:" + str(decay_chain.simulation_time_rom))
        decay_chain._plot_reduced_space(solution)
    if decay_chain.full_info == True:
        print(solution)

    decay_chain.solution_red = scipy.optimize.OptimizeResult(
        t=solution.t, y=decay_chain._expand(solution.y)
    )

    if decay_chain.info == True:
        decay_chain._plot_reduced_solution(
            decay_chain.solution_red, decay_chain.N_nuclides_red, log_scale
        )


def plot_reduced_space(solution) -> None:
    """Shows a plot of the simulation in reduced space for debugging purposes"""
    N_nuclides = solution.y.shape[0]
    for i in range(N_nuclides):
        plt.plot(solution.t, solution.y[i], label=str(i))
    if N_nuclides < 15:
        plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Nuclide amount")
    plt.title("Reduced space with " + str(N_nuclides) + " quasi-nuclides")
    plt.show()


def plot_reduced_solution(
    solution, quasi_nuclides: int, log_scale: bool = False
) -> None:
    """Shows a plot of the reduced order simulation for debugging purposes"""
    N_nuclides = solution.y.shape[0]
    for i in range(N_nuclides):
        plt.plot(solution.t, solution.y[i], label=str(i))
    if N_nuclides < 15:
        plt.legend()
    if log_scale == True:
        plt.semilogy()
    plt.xlabel("Time (s)")
    plt.ylabel("Nuclide amount")
    plt.title("Reduced order model based on " + str(quasi_nuclides) + " quasi-nuclides")
    plt.show()
