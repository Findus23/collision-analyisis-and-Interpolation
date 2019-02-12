from simulation_list import SimulationList

simulations = SimulationList.jsonlines_load()

for sim in simulations.simlist:
    print(vars(sim))
