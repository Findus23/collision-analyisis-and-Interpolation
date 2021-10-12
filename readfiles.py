from os import path
from pathlib import Path

from simulation import Simulation
from simulation_list import SimulationList

simulation_sets = {
    "winter": sorted(Path("../../tmp/winter/").glob("*"))
    # "original": sorted(glob("../data/*")),
    # "cloud": sorted(glob("../../Bachelorarbeit_data/results/*"))
    # "benchmark": sorted(glob("../../Bachelorarbeit_benchmark/results/*"))
}
simulations = SimulationList()

for set_type, directories in simulation_sets.items():
    for dir in directories:
        print(dir)
        spheres_file = dir / "spheres_ini.log"
        aggregates_file = sorted(dir.glob("frames/aggregates.*"))[-1]
        if not path.exists(spheres_file) or not path.exists(aggregates_file):
            print(f"skipping {dir}")
            continue
        sim = Simulation()
        sim.load_params_from_setup_txt(dir / "cfg.txt")
        sim.type = set_type
        sim.load_params_from_spheres_ini_log(spheres_file)
        sim.load_params_from_aggregates_txt(aggregates_file)
        # sim.assert_all_loaded()
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

simulations.jsonlines_save(Path("winter.jsonl"))
