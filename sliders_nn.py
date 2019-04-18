import matplotlib.pyplot as plt
import numpy as np
from keras.engine.saving import load_model
from matplotlib.collections import QuadMesh
from matplotlib.widgets import Slider
from sklearn.preprocessing import StandardScaler

from simulation_list import SimulationList

simlist = SimulationList.jsonlines_load()
train_data = simlist.simlist

X = np.array([[s.mcode, s.wpcode, s.wtcode, s.gammacode, s.alphacode, s.vcode] for s in train_data])
scaler = StandardScaler()
scaler.fit(X)




fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
t = np.arange(0.0, 1.0, 0.001)
mcode_default, gamma_default, wt_default, wp_default = [24.0, 1, 10.0, 10.0]


xrange = np.linspace(-0.5, 60.5, 100)
yrange = np.linspace(0.5, 5.5, 100)
xgrid, ygrid = np.meshgrid(xrange, yrange)
mcode = 24.
wpcode = 10
wtcode = 10
gammacode = 1

testinput = np.array([[mcode, wpcode, wtcode, gammacode, np.nan, np.nan]] * 100 * 100)
testinput[::, 4] = xgrid.flatten()
testinput[::, 5] = ygrid.flatten()
testinput = scaler.transform(testinput)

model = load_model("model.hd5")

testoutput = model.predict(testinput)
outgrid = np.reshape(testoutput, (100, 100))
mesh = plt.pcolormesh(xgrid, ygrid, outgrid, cmap="Blues", vmin=0, vmax=1)  # type:QuadMesh
plt.colorbar()


axcolor = 'lightgoldenrodyellow'
ax_mcode = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_gamma = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_wt = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_wp = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

s_mcode = Slider(ax_mcode, 'mcode', 21, 25, valinit=mcode_default)
s_gamma = Slider(ax_gamma, 'gamma', 0.1, 1, valinit=gamma_default)
s_wt = Slider(ax_wt, 'wt', 10, 20, valinit=wt_default)
s_wp = Slider(ax_wp, 'wp', 10, 20, valinit=wp_default)


def update(val):
    mcode = s_mcode.val
    gamma = s_gamma.val
    wt = s_wt.val
    wp = s_wp.val
    testinput = np.array([[mcode, wp, wt, gamma, np.nan, np.nan]] * 100 * 100)
    testinput[::, 4] = xgrid.flatten()
    testinput[::, 5] = ygrid.flatten()
    testinput = scaler.transform(testinput)

    testoutput = model.predict(testinput)
    outgrid = np.reshape(testoutput, (100, 100))
    # if not isinstance(datagrid, np.ndarray):
    #     return False
    formatedgrid = outgrid[:-1, :-1]

    mesh.set_array(formatedgrid.ravel())

    fig.canvas.draw_idle()


s_gamma.on_changed(update)
s_mcode.on_changed(update)
s_wp.on_changed(update)
s_wt.on_changed(update)

plt.show()
