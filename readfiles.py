from glob import glob
from os import path

from simulation import Simulation
from simulation_list import SimulationList

directories = sorted(glob("../data/*"))

simulations = SimulationList()

for dir in directories:
    spheres_file = dir + "/spheres_ini_log"
    aggregates_file = dir + "/sim/aggregates.txt"
    if not path.exists(spheres_file) or not path.exists(aggregates_file):
        print(f"skipping {dir}")
        continue
    if "id" not in dir:
        continue
    sim = Simulation()
    sim.load_params_from_dirname(path.basename(dir))
    sim.load_params_from_spheres_ini_log(spheres_file)
    sim.load_params_from_aggregates_txt(aggregates_file)
    sim.assert_all_loaded()
    simulations.append(sim)
    print(vars(sim))
    # exit()

print(len(simulations.simlist))

simulations.jsonlines_save()
