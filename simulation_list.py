import json
import pickle
from typing import List

import numpy as np

from simulation import Simulation


class SimulationList:
    simlist: List[Simulation]

    def __init__(self, simlist: list = None):
        if simlist is None:
            self.simlist = []
        else:
            self.simlist = simlist

    def append(self, value: Simulation):
        self.simlist.append(value)

    def pickle_save(self, filebasename="save"):
        with open(f'{filebasename}.pickle', 'wb') as file:
            pickle.dump(self.simlist, file)

    @classmethod
    def pickle_load(cls, filebasename="save"):
        with open(f'{filebasename}.pickle', 'rb') as file:
            return cls(pickle.load(file))

    def jsonlines_save(self, filebasename="save"):
        with open(f'{filebasename}.jsonl', 'w') as file:
            for sim in self.simlist:
                file.write(json.dumps(vars(sim)) + "\n")

    @classmethod
    def jsonlines_load(cls, filebasename="save"):
        simlist = cls()
        with open(f'{filebasename}.jsonl', 'r') as file:
            for line in file:
                sim = Simulation.from_dict(json.loads(line))
                simlist.append(sim)
        return simlist

    @property
    def as_matrix(self):
        entrylist = []
        for sim in self.simlist:
            if sim.type == "original":
                entrylist.append(
                    [sim.mcode, sim.wpcode, sim.wtcode, sim.gammacode, sim.alphacode, sim.vcode,
                     sim.water_retention_both]
                )
        return np.asarray(entrylist)

    @property
    def X(self):
        return np.array([
            [s.alpha, s.v, s.projectile_mass, s.gamma, s.target_water_fraction, s.projectile_water_fraction]
            for s in self.simlist
        ])

    @property
    def Y(self):
        return np.array([s.water_retention_both for s in self.simlist ])

