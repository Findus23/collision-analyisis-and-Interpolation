import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from simulation_list import SimulationList

mcode, gamma, wt, wp = [24.0, 1, 10.0, 10.0]
simlist = SimulationList.jsonlines_load()
simulations = simlist.simlist
print(simulations)

simsubset = [sim for sim in simulations if
             sim.mcode == mcode and sim.gammacode == gamma and sim.wtcode == wt and sim.wpcode == wp]

data = np.array([[s.alphacode, s.vcode] for s in simsubset])
values = np.array([s.water_retention_both for s in simsubset])

factor = 1

grid_x, grid_y = np.mgrid[-0.5:60.5:100j, 0.5:5.5:100j]
N = factor * np.arange(0., 1.01, 0.01)

print(grid_x)
grid_c = griddata(data, values, (grid_x, grid_y), method="linear")
print(grid_c)
plt.title("m={:3.0f}, gamma={:3.1f}, wt={:2.0f}, wp={:2.0f}\n".format(mcode, gamma, wt, wp))

# plt.contourf(grid_x, grid_y, grid_c, N, cmap="Blues", vmin=0, vmax=1)
plt.pcolormesh(grid_x, grid_y, grid_c, cmap="Blues", vmin=0, vmax=1)
plt.colorbar().set_label("test")

plt.scatter(data[:, 0], data[:, 1], c="black", cmap="Blues")
plt.savefig("vis.png")
plt.show()
