"""
Just a demo file on how to quickly read the dataset
"""

from simulation_list import SimulationList

simlist = SimulationList.jsonlines_load()

for s in simlist.simlist:
    if s.type == "original" or s.runid != 223:
        continue
    print(vars(s))
    if s.water_retention_both < 0:
        print(s.runid)
