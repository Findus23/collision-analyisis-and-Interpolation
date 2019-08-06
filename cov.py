import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from simulation_list import SimulationList

simulations = SimulationList.jsonlines_load()
np.set_printoptions(linewidth=1000, edgeitems=4)

x = simulations.as_matrix
labels = ["impact angle $\\alpha$", " collision speed $v$", "projectile mass", "mass fraction $\\gamma$",
          "target water fraction", "projectile water fraction"]
matrixlabels = labels + ["water retention"]

print(x)
print(x.shape)
corr = np.corrcoef(x.T)
print(np.cov(x.T))
X = np.copy(x)
X -= X.mean(axis=0)
manual = np.dot(X.T, X.conj()) / (x.shape[0] - 1)
assert np.allclose(manual, np.cov(x.T))
print(corr.shape)
simple_cov = corr[-1, :-1]
# plot correclation matrix
plt.matshow(corr)
# plt.xticks(range(len(matrixlabels)), matrixlabels, rotation=90)
# plt.yticks(range(len(matrixlabels)), matrixlabels)
# plt.colorbar()
# plt.savefig("correlation.pdf", transparent=True)
# plt.tight_layout()
# plt.show()
plt.close()

ax = plt.gca()  # type:Axes
print(len(labels), len(simple_cov))
print(simple_cov)
plt.barh(range(len(simple_cov)), simple_cov)
# ax.set_xticks(index + bar_width / 2)
ax.set_yticklabels([0] + labels)
ax2 = ax.twinx()  # type:Axes
ax2.set_yticklabels([0] + ["{:.2f}".format(a) for a in simple_cov])
ax2.set_ylim(ax.get_ylim())
plt.tight_layout()
plt.savefig("../arbeit/images/cov.pdf")
plt.show()
