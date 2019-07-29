import matplotlib.pyplot as plt
import numpy as np
# from scipy.spatial import voronoi_plot_2d, Delaunay
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.qhull import Delaunay
from matplotlib.axes import Axes

np.random.seed(10)

points = np.random.rand(100, 2)

testpoint = (0.7, 0.7)
fig1 = plt.figure()
ax = fig1.add_subplot()  # type:Axes
fig2 = plt.figure()
ax2 = fig2.add_subplot()  # type:Axes
fig3=plt.figure()
ax3d = fig3.add_subplot(projection='3d')  # type:Axes3D

# vor = Voronoi(grid)
#
#
# fig = voronoi_plot_2d(vor)
tri = Delaunay(points)
print(type(tri))
ax.scatter(points[:, 0], points[:, 1])
ax2.scatter(points[:, 0], points[:, 1])
#
center_tri = tri.find_simplex(np.array([testpoint]))[0]
print(center_tri)
print(tri.simplices[center_tri])

ax.scatter(*testpoint, color="green", zorder=2)
ax2.scatter(*testpoint, color="green", zorder=2)
close_points = []
for dot in tri.simplices[center_tri]:
    point = tri.points[dot]
    z = (point[0] - 0.5) ** 2 + (point[1] - 0.5) ** 2
    close_points.append([point[0], point[1], z])
    print(point)
    print("test")
    ax2.scatter(point[0], point[1], color="red", zorder=2)
    ax3d.scatter(point[0], point[1], z, color="red", zorder=2)
ax2.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
# plt.show()
# plt.close()
close_points = np.asarray(close_points)
z = (points[:, 0] - 0.5) ** 2 + (points[:, 1] - 0.5) ** 2

ax3d.scatter(points[:, 0], points[:, 1], z)

p1, p2, p3 = close_points

v1 = p3 - p1
v2 = p2 - p1

print(v1, v2)

# the cross product is a vector normal to the plane
cp = np.cross(v1, v2)
print(cp)
a, b, c = cp

# This evaluates a * x3 + b * y3 + c * z3 which equals d
d = np.dot(cp, p3)

x = np.linspace(0.5, .9, 5)
y = np.linspace(0.5, .9, 5)
X, Y = np.meshgrid(x, y)

Z = (d - a * X - b * Y) / c

z_center = (d - a * testpoint[0] - b * testpoint[1]) / c

ax3d.scatter(*testpoint, z_center, color="green")

ax3d.plot_surface(X, Y, Z, color="lightgreen", alpha=0.4)


fig1.savefig("../arbeit/images/vis2d1.pdf")
fig2.savefig("../arbeit/images/vis2d2.pdf")
fig3.savefig("../arbeit/images/vis2d3.pdf")
plt.show()
