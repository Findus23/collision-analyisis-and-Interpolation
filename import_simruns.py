from pathlib import Path

from simulation import Simulation
from simulation_list import SimulationList

simulations = SimulationList()

with open("RSMC_ML_dataset.txt") as f:
    for line in f:
        if line.startswith("#"):
            continue

        sim = Simulation()
        cols = line.split()
        sim.type = "simrun_import"
        rsmc_run, col_no, time, impact_vel, impact_vel_vesc, impact_angle, \
        mass_p, mass_t, mass_la, mass_sla, \
        wmf_p, wmf_t, wmf_la, wmf_sla, \
        cmf_p, cmf_t, cmf_la, cmf_sla = cols
        sim.runid = rsmc_run + " " + col_no
        sim.v = float(impact_vel_vesc)
        sim.alpha = float(impact_angle)
        sim.projectile_mass = float(mass_p)
        sim.target_mass = float(mass_t)
        sim.total_mass = sim.projectile_mass + sim.target_mass
        sim.largest_aggregate_mass = float(mass_la)
        if sim.largest_aggregate_mass < 0:
            sim.largest_aggregate_mass = 0
        sim.second_largest_aggregate_mass = float(mass_sla)
        if sim.second_largest_aggregate_mass < 0:
            sim.second_largest_aggregate_mass = 0

        sim.projectile_water_fraction = float(wmf_p)
        sim.target_water_fraction = float(wmf_t)
        sim.largest_aggregate_water_fraction = float(wmf_la)
        if sim.largest_aggregate_water_fraction < 0:
            sim.largest_aggregate_water_fraction = 0

        sim.second_largest_aggregate_water_fraction = float(wmf_sla)
        if sim.second_largest_aggregate_water_fraction < 0:
            sim.second_largest_aggregate_water_fraction = 0

        sim.projectile_core_fraction = float(cmf_p)
        sim.target_core_fraction = float(cmf_t)
        sim.largest_aggregate_core_fraction = float(cmf_la)
        if sim.largest_aggregate_core_fraction < 0:
            sim.largest_aggregate_core_fraction = 0

        sim.second_largest_aggregate_core_fraction = float(cmf_sla)
        if sim.second_largest_aggregate_core_fraction < 0:
            sim.second_largest_aggregate_core_fraction = 0

        if not sim.initial_water_mass or not sim.initial_core_mass:
            print(vars(sim))
            print(cols)
            # exit()
            continue
        simulations.append(sim)
        # if len(simulations.simlist)>1000:
        #     break
print(len(simulations.simlist))
simulations.jsonlines_save(Path("rsmc_dataset.jsonl"))
