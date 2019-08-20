from glob import glob
from os import path

from simulation import Simulation
from simulation_list import SimulationList

simulation_sets = {
    "original": sorted(glob("../data/*")),
    "cloud": sorted(glob("../../Bachelorarbeit_data/results/*"))
}
simulations = SimulationList()

for set_type, directories in simulation_sets.items():
    for dir in directories:
        original = set_type == "original"
        spheres_file = dir + "/spheres_ini_log"
        aggregates_file = dir + ("/sim/aggregates.txt" if original else "/aggregates.txt")
        if not path.exists(spheres_file) or not path.exists(aggregates_file):
            print(f"skipping {dir}")
            continue
        if "id" not in dir and original:
            continue
        sim = Simulation()
        if set_type == "original":
            sim.load_params_from_dirname(path.basename(dir))
        else:
            sim.load_params_from_json(dir + "/parameters.json")
        sim.type = set_type
        sim.load_params_from_spheres_ini_log(spheres_file)
        sim.load_params_from_aggregates_txt(aggregates_file)
        sim.assert_all_loaded()
        if sim.rel_velocity < 0 or sim.distance < 0:
            # Sometimes in the old dataset the second object wasn't detected.
            # To be save, we'll exclude them
            print(vars(sim))
            continue
        if sim.largest_aggregate_water_fraction < 0:
            # a handful of simulations had a typo in the aggregates simulations.
            # to fix those rerun aggregates on all of them
            print(vars(sim))
            raise ValueError("invalid aggregate data. Please rerun postprocessing")
        if sim.water_retention_both < 0 or sim.water_retention_both > 1:
            print(vars(sim))
            print(sim.water_retention_both)
            raise ValueError("water retention is invalid")
        simulations.append(sim)
        # print(vars(sim))

    print(len(simulations.simlist))

simulations.jsonlines_save()
