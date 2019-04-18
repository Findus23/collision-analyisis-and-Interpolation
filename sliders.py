import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.widgets import Slider
from scipy.interpolate import griddata

from simulation_list import SimulationList

simlist = SimulationList.jsonlines_load()
simulations = simlist.simlist

for sim in simulations:
    print(sim.gammacode)

grid_x, grid_y = np.mgrid[-0.5:60.5:100j, 0.5:5.5:100j]


def get_data(mcode, gamma, wt, wp):
    print([mcode, gamma, wt, wp])
    simsubset = [sim for sim in simulations if
                 sim.mcode == mcode and sim.gammacode == gamma and sim.wtcode == wt and sim.wpcode == wp]

    if len(simsubset) == 0:
        print("skip")
        return False

    print(len(simsubset))
    data = np.array([[s.alphacode, s.vcode] for s in simsubset])
    values = np.array([s.water_retention_both for s in simsubset])

    grid_c = griddata(data, values, (grid_x, grid_y), method="linear")
    return grid_c


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
t = np.arange(0.0, 1.0, 0.001)
mcode_default, gamma_default, wt_default, wp_default = [24.0, 1, 10.0, 10.0]

datagrid = get_data(mcode_default, gamma_default, wt_default, wp_default)

mesh = plt.pcolormesh(grid_x, grid_y, datagrid, cmap="Blues", vmin=0, vmax=1)  # type:QuadMesh
plt.colorbar()
print(type(mesh))

axcolor = 'lightgoldenrodyellow'
ax_mcode = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_gamma = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_wt = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_wp = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

s_mcode = Slider(ax_mcode, 'mcode', 21, 25, valinit=mcode_default, valstep=1)
s_gamma = Slider(ax_gamma, 'gamma', 0.1, 1, valinit=gamma_default, valstep=0.1)
s_wt = Slider(ax_wt, 'wt', 10, 20, valinit=wt_default, valstep=10)
s_wp = Slider(ax_wp, 'wp', 10, 20, valinit=wp_default, valstep=10)


def update(val):
    mcode = s_mcode.val
    gamma = s_gamma.val
    wt = s_wt.val
    wp = s_wp.val
    datagrid = get_data(mcode, gamma, wt, wp)
    if not isinstance(datagrid, np.ndarray):
        return False
    formatedgrid = datagrid[:-1, :-1]

    mesh.set_array(formatedgrid.ravel())

    fig.canvas.draw_idle()


s_gamma.on_changed(update)
s_mcode.on_changed(update)
s_wp.on_changed(update)
s_wt.on_changed(update)

plt.show()
