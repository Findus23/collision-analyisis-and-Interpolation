"""
This has nothing to do with the rest, but is just a very simple test if I understand pcolormesh correctly.
"""
import numpy as np
from matplotlib import pyplot as plt


def func(x, y):
    return (np.sin(y) + np.sin(x) + 1) / 3


xs = np.linspace(-15, 15, 30)
ys = np.linspace(-15, 15, 300)

res = []
for y in ys:
    rererere = []
    for x in xs:
        rererere.append(func(x, y))
    res.append(rererere)

data = np.asarray(res)

# plt.contourf(xs, ys, data, 100, cmap="Blues", vmin=0, vmax=1)
plt.pcolormesh(xs, ys, data, cmap="Blues", vmin=0, vmax=1)
plt.colorbar()

plt.show()
