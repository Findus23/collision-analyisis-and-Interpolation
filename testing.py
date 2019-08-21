import json
from statistics import mean

import numpy as np
from keras.engine.saving import load_model

from CustomScaler import CustomScaler
from config import water_fraction
from interpolators.griddata import GriddataInterpolator
from interpolators.rbf import RbfInterpolator
from simulation import Simulation
from simulation_list import SimulationList

simulations = SimulationList.jsonlines_load()

scaler = CustomScaler()
scaler.fit(simulations.X)
model = load_model("model.hd5" if water_fraction else "model_mass.hd5")


def squared_error(inter: float, correct: float) -> float:
    return (inter - correct) ** 2


def absolute_error(inter: float, correct: float) -> float:
    return abs(inter - correct)


def neural_network_test(scaled_input) -> float:
    nn_input = np.asarray([scaled_input])
    testoutput = model.predict(nn_input)[0][0]
    return testoutput


def rbf_test(scaled_parameters) -> float:
    scaled_data = scaler.transform_data(simulations.X)
    interpolator = RbfInterpolator(scaled_data, simulations.Y)
    result = interpolator.interpolate(*scaled_parameters)

    return result


def grid_test(scaled_parameters) -> float:
    scaled_data = scaler.transform_data(simulations.X)
    interpolator = GriddataInterpolator(scaled_data, simulations.Y)
    result = interpolator.interpolate(*scaled_parameters)

    return float(result)


nn_squared_errors = []
nn_errors = []
rbf_squared_errors = []
rbf_errors = []
grid_squared_errors = []
grid_errors = []
cachefile="grid-testing-cache.json" if water_fraction else "grid-testing-cache-mass.json"
try:
    with open(cachefile) as f:
        raw_data = json.load(f)
        grid_testing_cache = {int(key): value for key, value in raw_data.items()}
except FileNotFoundError:
    grid_testing_cache = {}
sim: Simulation
a = 0
for sim in simulations.simlist:
    if not sim.testcase:
        continue
    a += 1
    testinput = [sim.alpha, sim.v, sim.projectile_mass, sim.gamma,
                 sim.target_water_fraction, sim.projectile_water_fraction]
    scaled_input = list(scaler.transform_parameters(testinput))
    nn_output = neural_network_test(scaled_input)
    nn_squared_errors.append(squared_error(nn_output, sim.water_retention_both))
    nn_errors.append(absolute_error(nn_output, sim.water_retention_both))

    rbf_output = rbf_test(scaled_input)
    rbf_squared_errors.append(squared_error(rbf_output, sim.water_retention_both))
    rbf_errors.append(absolute_error(rbf_output, sim.water_retention_both))

    if sim.runid in grid_testing_cache:
        grid_output = grid_testing_cache[sim.runid]
    else:
        grid_output = grid_test(scaled_input)
        if np.isnan(grid_output):
            grid_output = False
        grid_testing_cache[sim.runid] = grid_output
        with open(cachefile, "w") as f:
            json.dump(grid_testing_cache, f)
    if grid_output:
        grid_squared_errors.append(squared_error(grid_output, sim.water_retention_both))
        grid_errors.append(absolute_error(grid_output, sim.water_retention_both))

    print(nn_output, rbf_output, grid_output, sim.water_retention_both)
print(a)
print()

# print(nn_squared_errors)
print(mean(nn_squared_errors))
print(mean(nn_errors))
print()
# print(rbf_squared_errors)
print(mean(rbf_squared_errors))
print(mean(rbf_errors))
print()

# print(grid_squared_errors)
print(mean(grid_squared_errors))
print(mean(grid_errors))
