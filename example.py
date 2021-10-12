"""
Just a demo file on how to quickly read the dataset
"""
from pathlib import Path
from pprint import pprint

from simulation_list import SimulationList


simlist = SimulationList.jsonlines_load(Path("save.jsonl"))

for s in simlist.simlist:
    if not s.testcase:
        continue
    if s.water_retention_both < 0.2:
        pprint(vars(s))
        print(s.water_retention_both)
