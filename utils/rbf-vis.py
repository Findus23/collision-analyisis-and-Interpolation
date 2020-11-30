import numpy as np

from matplotlib import pyplot as plt

np.random.seed(15)


def theta(r):
    return np.exp(-r ** 2)


points = [0, 3, 5]
values = [0.2, 0.8, 0.1]

# points = np.random.rand(15) * 5
# values = np.sin(points)

plt.scatter(points, values, label="data points")

subtract = np.zeros((len(points), len(points)))

for i in range(len(points)):
    for j in range(len(points)):
        subtract[i][j] = abs(points[i] - points[j])

print(subtract)

left_side = theta(subtract)
print(left_side)

lambdas = np.linalg.solve(left_side, values)

print(lambdas)


def s(x, i=None):
    running_sum = 0
    for index, lam in enumerate(lambdas):
        if i is None or i == index:
            running_sum += lam * theta(x - points[index])
    return running_sum


x = np.linspace(0, 5, 100)
y = s(x)

plt.plot(x, y, label="$s(x)$", zorder=-1)

for i in range(len(points)):
    plt.plot(x, s(x, i), label=f"RBF {i}",zorder=-2)
plt.legend()

# x_interpol = 2.3
# y_interpol = s(x_interpol)
# plt.scatter(x_interpol,y_interpol,color="green")

plt.savefig("../arbeit/images/rbf1.pdf")
# plt.savefig("../arbeit/images/rbf2.pdf")
plt.show()
