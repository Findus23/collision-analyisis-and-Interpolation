import json
import os
import pickle
from typing import List

import numpy as np

from config import water_fraction
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

    def save_path(self, extension):
        script_dir = os.path.dirname(__file__)
        rel_path = "save" + extension
        return os.path.join(script_dir, rel_path)

    def pickle_save(self):
        with open(self.save_path(".pickle"), 'wb') as file:
            pickle.dump(self.simlist, file)

    @classmethod
    def pickle_load(cls):
        tmp = cls()
        with open(cls.save_path(tmp, ".pickle"), 'rb') as file:
            return cls(pickle.load(file))

    def jsonlines_save(self):
        with open(self.save_path(".jsonl"), 'w') as file:
            for sim in self.simlist:
                file.write(json.dumps(vars(sim)) + "\n")

    @classmethod
    def jsonlines_load(cls):
        simlist = cls()
        with open(cls.save_path(simlist, ".jsonl"), 'r') as file:
            for line in file:
                sim = Simulation.from_dict(json.loads(line))
                simlist.append(sim)
        return simlist

    @property
    def as_matrix(self):
        entrylist = []
        for sim in self.simlist:
            if not sim.testcase:
                entrylist.append(
                    [sim.alpha, sim.v, sim.projectile_mass, sim.gamma, sim.target_water_fraction,
                     sim.projectile_water_fraction, sim.water_retention_both]
                )
        return np.asarray(entrylist)

    @property
    def X(self):
        return np.array([
            [s.alpha, s.v, s.projectile_mass, s.gamma, s.target_water_fraction, s.projectile_water_fraction]
            for s in self.simlist if not s.testcase
        ])

    @property
    def Y(self):
        return self.Y_water if water_fraction else self.Y_mass

    @property
    def Y_mass(self):
        return np.array([s.mass_retention_both for s in self.simlist if not s.testcase])

    @property
    def Y_water(self):
        return np.array([s.water_retention_both for s in self.simlist if not s.testcase])
