from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from simulation_list import SimulationList
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
simulations = SimulationList.jsonlines_load()
np.set_printoptions(linewidth=1000,edgeitems=4)
data = simulations.as_matrix
simpledata=data[:, 4:]
pca = PCA(n_components=3)
pca.fit(simpledata)
print(pca.components_)
X_pca = pca.transform(simpledata)
X_new = pca.inverse_transform(X_pca)
print(X_pca)


fig = plt.figure()
ax = Axes3D(fig)
# ax.scatter(simpledata[:, 0],simpledata[:, 1], simpledata[:, 2])
# plt.show()
ax.scatter(X_new[:, 0],X_new[:, 1], X_new[:, 2])
plt.show()
