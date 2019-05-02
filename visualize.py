import numpy as np
from matplotlib import pyplot as plt

from CustomScaler import CustomScaler
from interpolators.griddata import GriddataInterpolator
from simulation_list import SimulationList

mcode, gamma, wt, wp = [10**21, 1, 10.0, 10.0]
simlist = SimulationList.jsonlines_load()
simulations = simlist.simlist

data = np.array([[s.alphacode, s.vcode, 10**s.mcode, s.gammacode, s.wtcode, s.wpcode] for s in simulations])

scaler = CustomScaler()
scaler.fit(data)
scaled_data = scaler.transform_data(data)

values = np.array([s.water_retention_both for s in simulations])

alpharange = np.linspace(-0.5, 60.5, 100)
vrange = np.linspace(0.5, 5.5, 100)
grid_alpha, grid_v = np.meshgrid(alpharange, vrange)
interpolator = GriddataInterpolator(scaled_data, values)

# print(grid_alpha)
parameters = [grid_alpha, grid_v, mcode, gamma, wt, wp]
scaled_parameters = list(scaler.transform_parameters(parameters))

grid_result = interpolator.interpolate(*scaled_parameters)

print(grid_alpha.shape)
print(grid_result.shape)
plt.title("m={:3.0f}, gamma={:3.1f}, wt={:2.0f}, wp={:2.0f}\n".format(mcode, gamma, wt, wp))

# plt.contourf(grid_x, grid_y, grid_c, N, cmap="Blues", vmin=0, vmax=1)
plt.pcolormesh(grid_alpha, grid_v, grid_result, cmap="Blues", vmin=0, vmax=1)
plt.colorbar().set_label("water retention")
# plt.scatter(data[:, 0], data[:, 1], c=values, cmap="Blues")
plt.xlabel("impact angle $\\alpha$")
plt.ylabel("velocity $v$")
plt.tight_layout()
# plt.savefig("vis.png", transparent=True)
plt.show()
