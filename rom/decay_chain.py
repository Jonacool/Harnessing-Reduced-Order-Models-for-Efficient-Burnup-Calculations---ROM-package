""" 
Basic burnup analysis of a decay chain of N nuclides
written by: Jonathan Pilgram, november '23 - august '24
"""

import numpy as np
import time
import scipy
import matplotlib.pyplot as plt
import math
from .utils import signif


class DecayChain:
    """Class to simulate the decay process of a single chain of nuclides.

    Parameters
    ----------
    N_nuclides : int
        Amount of nuclides to simulate
    info : bool
        Controls whether to print extra information like plots and simulation time
    full_info : bool
        Shows even more information like snapshots and full solution

    Internal data
    -------------
    initial_composition : numpy zero array of size N_nuclides
        Used to set the initial composition of the nuclides
    decay_array : numpy zero array of size N_nuclides x N_nuclides
        The decay array used for the chain, starts as zeros array of size

    Initializes
    -----------
    solution : None
        The solution of the full model
    decay_array_red : None
        Comes from reduce_model.py
    initial_composition_red : None
        Comes from reduce_model.py
    rom_basis : None
        Comes from reduce_model.py
    simulation_time_rom : None
        Added by reduce_model.py
    decay_constants : None
        Added by build_decay_chain()
    """

    def __init__(
        self,
        N_nuclides: int,
        info: bool = False,
        full_info: bool = False,
    ) -> None:
        self.N_nuclides = N_nuclides
        self.info = info
        self.full_info = full_info
        self.initial_composition = np.zeros(self.N_nuclides)
        self.decay_array = np.zeros((self.N_nuclides, self.N_nuclides))
        self.decay_array_red = None
        self.initial_composition_red = None
        self.rom_basis = None
        self.solut = None

    # Should be rewritten I think to improve generality and remove disgusting for loops
    def _bateman_equation(self, t, N) -> list:
        """Private function: Returns the coupled bateman equations for use by scipy"""
        dNdt = np.zeros_like(N)
        for varying_nuclide in range(self.decay_array_temp.shape[0]):
            for nuclide in range(self.decay_array_temp.shape[1]):
                dNdt[varying_nuclide] += (
                    self.decay_array_temp[varying_nuclide, nuclide] * N[nuclide]
                )
        return dNdt

    def _run_simulation(
        self, decay_array: np.array, initial_composition: np.array, solver: str
    ):
        """Private function: wrapper for the scipy integrator"""
        self.decay_array_temp = decay_array

        return scipy.integrate.solve_ivp(
            self._bateman_equation,
            [0, self.sim_time],
            initial_composition,
            t_eval=np.linspace(0, self.sim_time, self.sim_steps + 1),
            method=solver,
        )

    def run_simulation(
        self, sim_time: int, sim_steps: int, solver: str = "Radau"
    ) -> None:
        """Runs the simulation using the solver of choice, default is Radau solver
        Other solvers are:
        'Euler_backward', 'Euler_forward', 'RK45'

        Saves the result in object as solution:
        solution.t - The list with evaluation times
        solution.y - 2D list with solutions for each nuclide

        paramters
        ---------
        sim_time : int
            The simulation time in seconds, saved to object
        sim_steps : int
            Amount of timesteps to simulate, saved to object
        solver : str (default: Radau)
            Solver of choice. All scipy solvers available and Euler_forward, Euler_backward, saved to object

        data saved
        ----------
        simulation_time_fom : float
            Simulation time for the full order solution in seconds
        """
        self.sim_time = sim_time
        self.sim_steps = sim_steps
        self.solver = solver
        if np.all(self.initial_composition == 0):
            print("Did you forget to set initial_composition?")

        start = time.time()

        if solver == "Euler_backward":
            solution = self._backward_euler(self.decay_array, self.initial_composition)
        else:
            solution = self._run_simulation(
                self.decay_array, self.initial_composition, solver
            )

        self.simulation_time_fom = time.time() - start
        if self.info == True:
            print("Simulation time:" + str(self.simulation_time_fom))
            self._plot_simulation(solution)
        if self.full_info == True:
            print(solution)
        self.solution = solution

    def _plot_simulation(self, solution) -> None:
        """Shows a plot of the decay simulation for debugging purposes"""
        N_nuclides = solution.y.shape[0]
        for i in range(N_nuclides):
            plt.plot(solution.t, solution.y[i], label=str(i))
        if N_nuclides < 15:
            plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Nuclide amount")
        plt.title("Full order decay chain with " + str(N_nuclides) + " nuclides")
        plt.show()

    def _backward_euler(self, decay_array: np.ndarray, initial_composition: np.ndarray):
        """Private function to run the simulation with first order backward euler

        parameters
        ----------
        Decay array: np.ndarray
            The decay array
        initial_composition: np.ndarray
            The initial composition

        Return
        ------
        solution
            Array containing the solution values"""
        nuclides_considered = np.shape(decay_array)[0]
        # Unsure how much dtype complex slows down calculation, could be 8 times
        result = np.zeros(
            (nuclides_considered, self.sim_steps + 1)
        )  # dtype = complex can be added
        result[:, 0] = initial_composition
        sim_times = np.linspace(0, self.sim_time, self.sim_steps + 1)
        x_old = result[:, 0]
        K_iter = (
            -decay_array
            + np.diag(np.ones(nuclides_considered)) * self.sim_steps / self.sim_time
        )
        for step in range(self.sim_steps):
            rhs = x_old * self.sim_steps / self.sim_time
            result[:, step + 1] = np.linalg.solve(K_iter, rhs)
            x_old = result[:, step + 1]

        solution = scipy.optimize.OptimizeResult(t=sim_times, y=result)
        return solution


def build_decay_chain(
    decay_chain,
    lambda_min: float,
    lambda_max: float,
    seed: int = None,
    terminate_stable: bool = True,
    significance: int = 7,
) -> None:
    """Builds an array with the decay constants of N nuclides in a chain, i.e. modifies the decay_array. Can terminate on  a stable nuclide.

    Parameters
    ----------
    lambda_min : float
        Minimal decay constant
    lambda_max : float
        Maximal value decay constant lambda
    seed (optional, default None) : int
        Used to set the seed for decay chain generation
    terminate_stable (default True): bool
       Whether to terminate on a stable nuclide.
    significance (default 7) : int
        Amount of signficant digits of constants; 7 is same as JEFF files

    Creates
    -------
    decay_constants : list
        The decay constants for the decay chain
    decay_array : np.array
        The decay array for the decay chain
    """

    np.random.seed(seed)
    decay_chain.decay_constants = signif(
        10
        ** (
            math.log10(lambda_min)
            + (math.log10(lambda_max / lambda_min))
            * np.random.rand(decay_chain.N_nuclides)
        ),
        significance,
    )

    if terminate_stable == True:
        decay_chain.decay_constants[-1] = 0

    decay_chain.decay_array = np.diag(decay_chain.decay_constants[:-1], -1) - np.diag(
        decay_chain.decay_constants
    )
