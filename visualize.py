import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from simulation_list import SimulationList

mcode, gamma, wt, wp = [22.0, 1, 10.0, 10.0]
simlist = SimulationList.jsonlines_load()
simulations = simlist.simlist

data = np.array([[s.alphacode, s.vcode, s.mcode, s.gammacode, s.wtcode, s.wpcode] for s in simulations])
values = np.array([s.water_retention_both for s in simulations])

factor = 1

alpharange = np.linspace(-0.5, 60.5, 100)
vrange = np.linspace(0.5, 5.5, 100)
grid_alpha, grid_v = np.meshgrid(alpharange, vrange)

N = factor * np.arange(0., 1.01, 0.01)

# print(grid_alpha)
grid_result = griddata(data, values, (grid_alpha, grid_v, mcode, gamma, wt, wp), method="nearest")
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
