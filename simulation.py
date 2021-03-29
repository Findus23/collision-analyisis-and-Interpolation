import json
from typing import Optional


class Simulation:

    def __init__(self):
        self.runid = None  # type:int
        self.vcode = None
        self.alphacode = None
        self.mcode = None
        self.gammacode = None
        self.wtcode = None
        self.wpcode = None
        self.type = None  # simulation set
        self.v = None  # v/v_esc
        self.alpha = None  # impact angle
        self.total_mass = None  # m
        self.projectile_mass = None  # mp
        self.target_mass = None  # mt
        self.projectile_water_fraction = None  # wp
        self.projectile_core_fraction = None
        self.target_water_fraction = None  # wt
        self.target_core_fraction = None
        self.largest_aggregate_mass = None  # mS1
        self.largest_aggregate_water_fraction = None  # wmfS1
        self.largest_aggregate_core_fraction = None
        self.second_largest_aggregate_mass = None  # mS2
        self.second_largest_aggregate_water_fraction = None  # wmfS2
        self.second_largest_aggregate_core_fraction = None
        self.rel_velocity = None  # vrel
        self.rel_velocity_per_esc_velocity = None  # vrel_over_vesc
        self.desired_N = None
        self.actual_N = None
        self.relaxation_time = None
        self.miluphcuda_time = None
        self.setup_time = None

    @classmethod
    def from_dict(cls, data: dict):
        sim = cls()
        for key in data:
            setattr(sim, key, data[key])
        return sim

    @property
    def gamma(self) -> float:
        return self.projectile_mass / self.target_mass

    @property
    def relative_projectile_mass(self) -> float:
        return self.projectile_mass / self.total_mass

    @property
    def relative_target_mass(self) -> float:
        return self.target_mass / self.total_mass

    @property
    def largest_aggregate_relative_mass(self) -> float:
        """
        p['mS1_over_mt'] = p['mS1'] / p['mt']
        """
        return self.largest_aggregate_mass / self.target_mass

    @property
    def largest_aggregate_mantle_fraction(self) -> float:
        return 1 - self.largest_aggregate_core_fraction - self.largest_aggregate_water_fraction

    @property
    def second_largest_aggregate_relative_mass(self) -> float:
        """
        p['mS2_over_mp'] = p['mS2'] / p['mp']
        """
        return self.second_largest_aggregate_mass / self.projectile_mass

    @property
    def second_largest_aggregate_mantle_fraction(self) -> float:
        return 1 - self.second_largest_aggregate_core_fraction - self.second_largest_aggregate_water_fraction

    @property
    def projectile_mantle_fraction(self):
        return 1 - self.projectile_water_fraction - self.projectile_core_fraction

    @property
    def target_mantle_fraction(self):
        return 1 - self.target_water_fraction - self.target_core_fraction

    @property
    def initial_water_mass(self) -> float:
        return self.projectile_mass * self.projectile_water_fraction + self.target_mass * self.target_water_fraction

    @property
    def initial_core_mass(self) -> float:
        return self.projectile_mass * self.projectile_core_fraction + self.target_mass * self.target_core_fraction

    @property
    def initial_mantle_mass(self) -> float:
        return self.projectile_mass * self.projectile_mantle_fraction + self.target_mass * self.target_mantle_fraction

    @property
    def water_retention_both(self) -> float:
        """
        p['wretentionB'] = (p['mS1'] * p['wmfS1'] + p['mS2'] * p['wmfS2']) / (p['mp'] * p['wp'] + p['mt'] * p['wt'])
        """
        return (
                       self.largest_aggregate_mass * self.largest_aggregate_water_fraction
                       + self.second_largest_aggregate_mass * self.second_largest_aggregate_water_fraction
               ) / self.initial_water_mass

    @property
    def core_retention_both(self) -> float:
        return (
                       self.largest_aggregate_mass * self.largest_aggregate_core_fraction
                       + self.second_largest_aggregate_mass * self.largest_aggregate_core_fraction
               ) / self.initial_core_mass

    @property
    def mantle_retention_both(self) -> float:
        return (
                       self.largest_aggregate_mass * self.largest_aggregate_mantle_fraction
                       + self.second_largest_aggregate_mass * self.largest_aggregate_mantle_fraction
               ) / self.initial_mantle_mass

    @property
    def water_retention_main(self) -> float:
        """
        p['wretention1'] = p['mS1'] * p['wmfS1'] / (p['mp'] * p['wp'] + p['mt'] * p['wt'])
        """
        return self.largest_aggregate_mass * self.largest_aggregate_water_fraction / self.initial_water_mass

    @property
    def output_mass_fraction(self) -> Optional[float]:
        if not self.largest_aggregate_mass:
            return 0  # FIXME
        return self.second_largest_aggregate_mass / self.largest_aggregate_mass

    @property
    def original_simulation(self) -> bool:
        return self.type == "original"

    @property
    def testcase(self) -> bool:
        return self.type == "cloud" and 529 <= self.runid <= 631

    @property
    def simulation_key(self):
        return "id{:04d}_v{:.1f}_a{:.0f}_m{:.0f}_g{:.1f}_wt{:.1f}_wp{:.1f}".format(
            self.runid, self.vcode, self.alphacode, self.mcode, self.gammacode, self.wtcode, self.wpcode
        )

    def __repr__(self):
        return f"<Simulation '{self.simulation_key}'>"

    def load_params_from_dirname(self, dirname: str) -> None:
        params = dirname.split("_")
        self.runid = int(params[0][2:])
        self.vcode = float(params[1][1:])
        self.alphacode = float(params[2][1:])
        self.mcode = float(params[3][1:])
        self.gammacode = float(params[4][1:])
        self.wtcode = float(params[5][2:])
        self.wpcode = float(params[6][2:])
        assert dirname == self.simulation_key

    def load_params_from_json(self, paramfile: str) -> None:
        with open(paramfile) as f:
            data = json.load(f)
        self.runid = int(data["run_id"])
        self.vcode = data["v_code"]
        self.alphacode = data["alpha_code"]
        self.mcode = data["m_code"]
        self.gammacode = data["gamma_code"]
        self.wtcode = data["wt_code"]
        self.wpcode = data["wp_code"]

    def load_params_from_spheres_ini_log(self, filename: str) -> None:
        with open(filename) as f:
            lines = [line.rstrip("\n") for line in f]
        for i in range(len(lines)):
            line = lines[i]
            if "Geometry:" in line:
                self.v = float(lines[i + 2].split(" = ")[-1])
                print(lines[i + 3])
                # self.alpha = float(lines[i + 3].split(" = ")[-1][:-1]) #for old format
                self.alpha = float(lines[i + 3].split(" = ")[-1].split(" ")[0])
            if "Masses:" in line:
                self.total_mass = float(lines[i + 2].split()[3])
                self.projectile_mass = float(lines[i + 4].split()[3])
                self.target_mass = float(lines[i + 6].split()[3])
            if "Mantle/shell mass fractions:" in line:
                self.projectile_water_fraction = float(lines[i + 1].split()[7])
                self.target_water_fraction = float(lines[i + 3].split()[7])
            if "Particle numbers" in line:
                self.desired_N = int(lines[i + 1].split()[3])
                self.actual_N = int(lines[i + 1].split()[-1])

    def load_params_from_aggregates_txt(self, filename: str) -> None:
        with open(filename) as f:
            lines = [line.rstrip("\n") for line in f]
        for i in range(len(lines)):
            line = lines[i]
            if "# largest aggregate" in line:
                self.largest_aggregate_mass = float(lines[i + 2].split()[0])
                self.largest_aggregate_water_fraction = float(lines[i + 2].split()[2])
            if "# 2nd-largest aggregate:" in line:
                self.second_largest_aggregate_mass = float(lines[i + 2].split()[0])
                self.second_largest_aggregate_water_fraction = float(lines[i + 2].split()[2])
            if "#    distance" in line:  # TODO: not sure if correct anymore
                self.distance = float(lines[i + 1].split()[0])
                self.rel_velocity = float(lines[i + 1].split()[1])
                self.rel_velocity_per_esc_velocity = float(lines[i + 1].split()[2])

    def load_params_from_pythontiming_json(self, filename: str) -> None:
        with open(filename) as f:
            data = json.load(f)
            self.miluphcuda_time = data["miluphcuda"]
            self.relaxation_time = data["relaxation"]
            self.setup_time = data["setup"]

    def assert_all_loaded(self) -> None:
        for key, value in self.__dict__.items():
            print(key, value)
            assert value is not None
