import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
np.set_printoptions(linewidth=1000, edgeitems=4)


def print_heading(text):
    print(f" {text} ".center(80, "-"))


data = np.loadtxt("data.txt")
labels = ["score", "age", "view_count", "body_length", "answer_count", "comment_count", "favourite_count"]
print_heading("raw data")  ############################

print(data)
print_heading("scaled")  ############################

scaler = StandardScaler()
scaler.fit(data)
x = scaler.transform(data)
# x=data
print(x)
n = 7
pca = PCA(n_components=n)
pca.fit(x)
print_heading("components")  ############################
print(pca.components_.shape)  # eigenvectors of covariance matrix
print(pca.components_)
print_heading("explained_variance")  ############################

print(pca.explained_variance_)  # n largest eigenvalues of covariance matrix
print(pca.explained_variance_ratio_, "(as ratio)")
print_heading("covariance")  ############################

print(pca.get_covariance().shape)  # eigenvectors
print(pca.get_covariance())

print_heading("transformed")  ############################

x_new = pca.transform(x)
print(x_new.shape)
print(x_new)
print_heading("inverse transformed and undone scale")  ############################

x_simple = scaler.inverse_transform(pca.inverse_transform(x_new))

print(x_simple.shape)
print(x_simple)

print(pca.explained_variance_)

plt.scatter(data[::, 0], data[::, 4], s=1)
plt.scatter(x_simple[::, 0], x_simple[::, 4], s=1)
plt.show()

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

# plot correclation matrix
cov = pca.get_covariance()
plt.matshow(cov)
plt.xticks(range(len(labels)), labels,rotation=90)
plt.yticks(range(len(labels)), labels)
plt.colorbar()

plt.show()
