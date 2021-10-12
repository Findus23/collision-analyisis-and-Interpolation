import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import QuadMesh
from matplotlib.widgets import Slider

from CustomScaler import CustomScaler
from network import Network

resolution = 100

with open("pytorch_model.json") as f:
    data = json.load(f)
    scaler = CustomScaler()
    scaler.means = np.array(data["means"])
    scaler.stds = np.array(data["stds"])

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
t = np.arange(0.0, 1.0, 0.001)
mcode_default, gamma_default, wt_default, wp_default = [24.0, 1, 15.0, 15.0]

alpharange = np.linspace(0, 60, resolution)
vrange = np.linspace(0.5, 5.5, resolution)
grid_alpha, grid_v = np.meshgrid(alpharange, vrange)

model = Network()
model.load_state_dict(torch.load("pytorch_model.zip"))

datagrid = np.zeros_like(grid_alpha)

mesh = plt.pcolormesh(grid_alpha, grid_v, datagrid, cmap="Blues", vmin=0, vmax=1, shading="auto")  # type:QuadMesh
plt.colorbar()

axcolor = 'lightgoldenrodyellow'
ax_mcode = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_gamma = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_wt = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_wp = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_mode = plt.axes([0.25, 0.05, 0.65, 0.03])

s_mcode = Slider(ax_mcode, 'mcode', 21, 25, valinit=mcode_default)
s_gamma = Slider(ax_gamma, 'gamma', 0.1, 1, valinit=gamma_default)
s_wt = Slider(ax_wt, 'wt', 1e-5, 1e-3, valinit=wt_default)
s_wp = Slider(ax_wp, 'wp', 1e-5, 1e-3, valinit=wp_default)
s_mode = Slider(ax_mode, 'shell/mantle/core/mass_fraction', 1, 4, valinit=1, valstep=1)


def update(val):
    mcode = s_mcode.val
    gamma = s_gamma.val
    wt = s_wt.val
    wp = s_wp.val
    mode = s_mode.val
    testinput = np.array([[np.nan, np.nan, 10 ** mcode, gamma, wt, wp]] * resolution * resolution)
    testinput[::, 0] = grid_alpha.flatten()
    testinput[::, 1] = grid_v.flatten()
    testinput = scaler.transform_data(testinput)

    try:
        testoutput: torch.Tensor = model(torch.from_numpy(testinput).to(torch.float))
        data = testoutput.detach().numpy()
        print(data.shape)
    except TypeError:  # can't convert np.ndarray of type numpy.object_.
        data = np.zeros((resolution ** 2, 3))

    datagrid = np.reshape(data[::, mode - 1], (resolution, resolution))

    mesh.set_array(datagrid.ravel())

    fig.canvas.draw_idle()


update(None)

s_gamma.on_changed(update)
s_mcode.on_changed(update)
s_wp.on_changed(update)
s_wt.on_changed(update)
s_mode.on_changed(update)

plt.show()
