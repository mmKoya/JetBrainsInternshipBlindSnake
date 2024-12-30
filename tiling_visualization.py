"""
Script for visualizing tiling when using slices between two curves y=S_1/x and y=S_2/y.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 500)


def create_fill_slice(area_low=100.0, area_high=200.0, resolution=100):
    x_low = np.geomspace(1, area_low, resolution)
    y_low = area_low / x_low

    curve_low = np.column_stack((x_low, y_low))
    curve_low = np.vstack((np.array([[0, area_low]]), curve_low, np.array([[area_low, 0]])))

    x_high = np.geomspace(1, area_high, resolution)
    y_high = area_high / x_high

    curve_high = np.column_stack((x_high, y_high))
    curve_high = np.vstack((np.array([[0, area_high]]), curve_high, np.array([[area_high, 0]])))

    return curve_low, curve_high


def tile_slice(area_low=100.0, area_high=200.0, resolution=100, w=10, h=10, tiling_resolution=5, color='k'):
    curve_low, curve_high = create_fill_slice(area_low, area_high, resolution)
    curve = np.vstack((curve_low, curve_high[::-1]))

    for x_ in range(-tiling_resolution, tiling_resolution + 1):
        for y_ in range(-tiling_resolution, tiling_resolution + 1):
            translated_curve = curve + np.array([[x_*w, y_*h]])

            plt.fill(translated_curve[:, 0], translated_curve[:, 1], color=color, alpha=0.5)

    return area_high * (1 + np.log(area_high)) - area_low * (1 + np.log(area_low))


w = 100
h = 100

plt.rcParams['figure.figsize'] = [8, 8]


n = np.arange(1, 6)
max_area = 10000
x = (-n + np.sqrt(n ** 2 + 4)) / 2
S = (n - 1 + x) * x * max_area

for s in S:
    tile_slice(s, max_area, w=w, h=h, tiling_resolution=5, resolution=100)
    plt.xlim([0, w])
    plt.ylim([0, h])

    plt.show()

