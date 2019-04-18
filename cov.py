import numpy as np
from matplotlib import pyplot as plt

from simulation_list import SimulationList

simulations = SimulationList.jsonlines_load()
np.set_printoptions(linewidth=1000, edgeitems=4)

x = simulations.as_matrix
labels = ["$M$", "$wp$", "$wt$", "$\\gamma$", "$\\alpha$", "$v$", "water"]
x[::, 0] = 10 ** x[::, 0]

print(x)
print(x.shape)
corr = np.corrcoef(x.T)
print(np.cov(x.T))
X = np.copy(x)
X -= X.mean(axis=0)
manual = np.dot(X.T, X.conj()) / (x.shape[0] - 1)
assert np.allclose(manual, np.cov(x.T))
print(corr)
simple_cov = corr[-1, :-1]
# plot correclation matrix
plt.matshow(corr)
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
plt.colorbar()
plt.savefig("correlation.pdf",transparent=True)
plt.show()
# plt.close()
plt.bar(range(len(simple_cov)), simple_cov)
# ax.set_xticks(index + bar_width / 2)
ax = plt.gca()
ax.set_xticklabels(labels)
plt.show()
