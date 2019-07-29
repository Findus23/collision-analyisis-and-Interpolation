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
        if sim.rel_velocity<0 or sim.distance<0:
            print(vars(sim))
            raise ValueError("invalid aggregate data. Please rerun postprocessing")
        simulations.append(sim)
        # print(vars(sim))

    print(len(simulations.simlist))

simulations.jsonlines_save()
