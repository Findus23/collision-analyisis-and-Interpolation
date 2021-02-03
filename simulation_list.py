import json
import pickle
from pathlib import Path
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

    def pickle_save(self, pickle_file: Path):
        with pickle_file.open("wb") as file:
            pickle.dump(self.simlist, file)

    @classmethod
    def pickle_load(cls, pickle_file: Path):
        tmp = cls()
        with pickle_file.open("rb") as file:
            return cls(pickle.load(file))

    def jsonlines_save(self, jsonl_file: Path):
        with jsonl_file.open("w") as file:
            for sim in self.simlist:
                file.write(json.dumps(vars(sim)) + "\n")

    @classmethod
    def jsonlines_load(cls, jsonl_file: Path):
        simlist = cls()
        with jsonl_file.open() as file:
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
