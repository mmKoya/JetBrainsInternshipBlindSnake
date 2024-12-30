"""
Plots ratios of used moves over given area for pathing algorithms. Graph should never cross 35 as to ensure the
constraint of 35*S maximum moves.
"""

from matplotlib import pyplot as plt
from pathing import FillPathing, FillPathingOptimized
from pathing import calc_bounds


def visualize_moves_over_area_ratios(mapping, path):
    ratio = []
    for k, v in mapping.items():
        ratio.append(v / k)

    sorted_areas = sorted(mapping.keys())

    ratios = [mapping[key] / key for key in sorted_areas]

    plt.plot(sorted_areas, ratios, color='r')
    plt.xscale("log")
    plt.xlabel("Area")
    plt.ylabel("Number of moves over area")

    plt.grid(True)
    plt.savefig(path, dpi=300)
    plt.show()


max_area = 1_000_000

path = FillPathing(35, max_area)

absolute_pos = [1, 1]
n_moves = 0
mapping = {}
targets = calc_bounds(max_area)
target_moves = sum(targets) - 1
while True:
    n_moves += 1
    direction = path.get_direction()
    absolute_pos[0] += direction[0]
    absolute_pos[1] += direction[1]
    area = absolute_pos[0] * absolute_pos[1]
    area = abs(area)
    if area != 0 and area <= max_area:
        target_moves -= 1
        if area in mapping:
            mapping[area] = max(mapping[area], n_moves+1)
        else:
            mapping[area] = n_moves+1

    if target_moves == 0:
        print(n_moves)
        break

visualize_moves_over_area_ratios(mapping, "FillPathing.png")


path = FillPathingOptimized(35, max_area, [132, 1217, 7583, 36244, 140538, 457739])

absolute_pos = [1, 1]
n_moves = 0
mapping = {}
target_moves = max_area - 1
while True:

    n_moves += 1
    direction = path.get_direction()
    if path.next_iter:
        absolute_pos = [1, 1]
    if not direction:
        print(n_moves)
        break
    absolute_pos[0] += direction[0]
    absolute_pos[1] += direction[1]

    if absolute_pos[0] not in mapping:
        target_moves -= 1
        mapping[absolute_pos[0]] = n_moves

    if target_moves == 0:
        print(n_moves)
        break

visualize_moves_over_area_ratios(mapping, "FillPathingOptimized.png")
