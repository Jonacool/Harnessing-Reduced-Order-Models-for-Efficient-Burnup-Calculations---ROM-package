""" 
written by: Jonathan Pilgram, november '23 - august '24
"""

import numpy as np
import rom
import matplotlib.pyplot as plt


# Stap niet in het valletje van de god-class. Laat dit als een teken bestaan van een pad dat je niet moet inslaan!
def auto_reduce_decaychain(
    nuclides_FOM: int,
    nuclides_ROM: int,
    lambda_min: float,
    lambda_max: float,
    sim_time: float,
    sim_steps: int,
    N_snaps: int,
):
    """Automatically runs a whole order reduction procedure for a nuclide decay chain

    Parameters
    ----------
    nuclides_FOM: int
        Amount of nuclides to simulate in the Full Order Model
    nuclides_ROM: int
        Amount of nuclides to run the Reduced Order Model with
    lambda_min: float
        Minimal decay constant possible for a nuclide
    lambda_max: float
        Maximal decay constant possible for a nuclide
    sim_time: float
        Time in seconds to simulate
    sim_steps: int
        Amount of simulation steps in the time window
    N_snaps: int
        Amount of snapshots taken to build the ROM"""

    FOM = rom.DecayChain(nuclides_FOM)
    FOM.build_decay_chain(lambda_min, lambda_max)
    FOM.initial_composition[0] = 1
    sol_fom = FOM.run_simulation(sim_time, sim_steps)
    snaps = rom.SnapshotMatrix(sol_fom)
    snaps.preheat_scipy_sol()
    snapshots = snaps.spatial_snapshot_matrix(N_snaps)
    reduction = rom.ReduceModel(snapshots)
    rom_basis = reduction.reduce_SVD(nuclides_ROM)
    ROM = rom.DecayChain(nuclides_ROM)
    ROM.decay_array = rom_basis.T @ FOM.decay_array @ rom_basis
    ROM.initial_composition = rom_basis.T @ FOM.initial_composition
    sol_red = ROM.run_simulation(sim_time, sim_steps)
    sol_red.y = rom_basis @ sol_red.y
    for i in range(nuclides_FOM):
        plt.plot(sol_red.t, sol_red.y[i], label=str(i))
    if nuclides_FOM < 15:
        plt.legend()
    plt.title("Reduced order decay chain with " + str(nuclides_ROM) + " nuclides")
    plt.show()
    stats = rom.Analysis(sol_fom, sol_red)
    print("Speedup factor:" + str(FOM.simulation_time / ROM.simulation_time))
    stats.print_error()
