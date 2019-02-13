from simulation import Simulation
from simulation_list import SimulationList

simulations = SimulationList.jsonlines_load()

sim: Simulation
for sim in simulations.simlist:
    print(sim.simulation_key)
