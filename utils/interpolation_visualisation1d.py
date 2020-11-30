import matplotlib.pyplot as plt
import numpy as np

np.random.seed(15)

fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 20]})
# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_ylim(0, 1)
ax1.xaxis.set_ticks_position('bottom')

points = np.sort(np.random.rand(20))
print(points)
testpoint = 0.4

ax1.scatter(points, np.zeros_like(points) + 0.5)
ax1.scatter(testpoint, 0.5, color="green")

for i, val in enumerate(points):
    if val > testpoint:
        next_i = i
        prev_i = i - 1
        break

Y = np.sin(points * 6)

ax2.scatter(points, Y)
ax2.scatter(points[prev_i], Y[prev_i], color="red")
ax1.scatter(points[prev_i], 0.5, color="red")
ax2.scatter(points[next_i], Y[next_i], color="red")
ax1.scatter(points[next_i], 0.5, color="red")

a, b = np.polyfit([points[prev_i], points[next_i]], [Y[prev_i], Y[next_i]], 1)
print(a, b)

linex = np.linspace(0.3, 0.6, 2)
liney = b + a * linex

testy = b + a * testpoint

ax2.plot(linex, liney, color="lightgreen", zorder=-2)
ax2.scatter(testpoint, testy, color="green")
plt.tight_layout()
plt.savefig("../arbeit/images/vis1d.pdf")
plt.show()
