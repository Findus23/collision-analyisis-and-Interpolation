from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.widgets import Slider, Button

from CustomScaler import CustomScaler
from interpolators.rbf import RbfInterpolator
from simulation_list import SimulationList

simlist = SimulationList.jsonlines_load(Path("rsmc_dataset.jsonl"))
resolution = 100

data = simlist.X
values = simlist.Y_mass_fraction

scaler = CustomScaler()
scaler.fit(data)
scaled_data = scaler.transform_data(data)
interpolator = RbfInterpolator(scaled_data, values)

alpharange = np.linspace(-0.5, 60.5, resolution)
vrange = np.linspace(0.5, 5.5, resolution)
grid_alpha, grid_v = np.meshgrid(alpharange, vrange)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
t = np.arange(0.0, 1.0, 0.001)
mcode_default, gamma_default, wt_default, wp_default = [24.0, 1, 15.0, 15.0]

datagrid = np.zeros_like(grid_alpha)

mesh = plt.pcolormesh(grid_alpha, grid_v, datagrid, cmap="Blues", vmin=0, vmax=1, shading="nearest")  # type:QuadMesh
plt.colorbar()

# axcolor = 'lightgoldenrodyellow'
ax_mcode = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_gamma = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_wt = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_wp = plt.axes([0.25, 0.25, 0.65, 0.03])
buttonax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(buttonax, 'Update', hovercolor='0.975')
# thetext = ax.text(-10, 0, "hello", fontsize=12) #type:Text

s_mcode = Slider(ax_mcode, 'mcode', 21, 25, valinit=mcode_default)
s_gamma = Slider(ax_gamma, 'gamma', 0.1, 1, valinit=gamma_default)
s_wt = Slider(ax_wt, 'wt', 1e-5, 1e-3, valinit=wt_default)
s_wp = Slider(ax_wp, 'wp', 1e-5, 1e-3, valinit=wp_default)


def update(val):
    print("start updating")
    # thetext.set_text("updating")
    # fig.canvas.draw()

    mcode = s_mcode.val
    gamma = s_gamma.val
    wt = s_wt.val
    wp = s_wp.val
    parameters = [grid_alpha, grid_v, 10 ** mcode, gamma, wt, wp]
    scaled_parameters = list(scaler.transform_parameters(parameters))

    datagrid = interpolator.interpolate(*scaled_parameters)
    print(datagrid)
    print(np.isnan(datagrid).sum() / (resolution * resolution))
    if not isinstance(datagrid, np.ndarray):
        return False

    mesh.set_array(datagrid.ravel())
    print("finished updating")
    # thetext.set_text("finished")

    fig.canvas.draw_idle()


update(None)

# s_gamma.on_changed(update)
# s_mcode.on_changed(update)
# s_wp.on_changed(update)
# s_wt.on_changed(update)
button.on_clicked(update)

plt.show()
