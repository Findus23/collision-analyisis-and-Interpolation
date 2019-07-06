import numpy as np

from CustomScaler import CustomScaler
from interpolators.griddata import GriddataInterpolator
from simulation_list import SimulationList

simlist = SimulationList.jsonlines_load()

data = np.array([
    [s.alpha, s.v, s.projectile_mass, s.gamma, s.target_water_fraction, s.projectile_water_fraction]
    for s in simlist.simlist if s.type == "original"
])
values = np.array([s.water_retention_both for s in simlist.simlist if s.type == "original"])

scaler = CustomScaler()
scaler.fit(data)
scaled_data = scaler.transform_data(data)
# interpolator = RbfInterpolator(scaled_data, values)
interpolator = GriddataInterpolator(scaled_data, values)

print("\t".join(["runid", "interpolated value", "real value", "alpha", "v", "m", "gamma"]))
for sim in simlist.simlist:
    if sim.type == "original" or sim.v > 10 or int(sim.runid) < 10:
        continue
    print(int(sim.runid))
    parameters = [sim.alpha, sim.v, sim.projectile_mass, sim.gamma, sim.target_water_fraction,
                  sim.projectile_water_fraction]
    scaled_parameters = list(scaler.transform_parameters(parameters))

    result = interpolator.interpolate(*scaled_parameters)
    print("\t".join(map(str, [sim.runid, result, sim.water_retention_both, *parameters[:-2]])))
    # exit()
