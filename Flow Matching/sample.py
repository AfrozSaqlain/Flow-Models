import numpy as np
import matplotlib.pyplot as plt

N = 1000
x_min, x_max = -4, 4
y_min, y_max = -4, 4
resolution = 100

x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

length = 4
checkboard = np.indices((length, length)).sum(axis=0) % 2

sampled_points = []
while len(sampled_points) < N:
    x_sample = np.random.uniform(x_min, x_max)
    y_sample = np.random.uniform(y_min, y_max)

    i = int((x_sample - x_min) / (x_max - x_min) * length)
    j = int((y_sample - y_min) / (y_max - y_min) * length)

    if checkboard[j, i] == 1:
        sampled_points.append((x_sample, y_sample))

sampled_points = np.array(sampled_points)

plt.figure(figsize=(6, 6))
plt.scatter(sampled_points[:, 0], sampled_points[:, 1], s=5, alpha=0.6)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title("Sampled Points (Checkerboard Mask)")
plt.gca().set_aspect('equal')
plt.show()
