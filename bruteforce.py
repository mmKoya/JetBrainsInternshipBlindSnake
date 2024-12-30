"""
Script for finding linear solutions by brute-force approach.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch
from utils import does_fill_batch


def find_path(area, mask=None, batch_size=2**16):
    # os.makedirs(f"{area}", exist_ok=True)

    batch = []
    current_sample = 0
    root = int(area**0.5)
    c = 0

    def path_from(positions: np.ndarray, i, j, mask=None):

        nonlocal c
        nonlocal batch
        nonlocal current_sample

        if mask is not None and mask[i, j] == 0:
            c += 1
            return []


        if i + j < area - root:
            c += 1
            return []

        if i + j == area:
            c += 1
            return []

        if i == 0 and j == area - 1:

            c += 1

            batch.append(positions)
            if len(batch) == batch_size:
                batch = np.array(batch)
                tensor = torch.tensor(batch, dtype=torch.bool, device='cuda')
                fills = does_fill_batch(tensor, area)
                batch = []
                if torch.any(fills):

                    boards = tensor[fills].cpu().numpy()

                    for board in boards:

                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(board)
                        ax.xaxis.set_major_locator(MultipleLocator(1))
                        ax.yaxis.set_major_locator(MultipleLocator(1))

                        plt.grid(True)
                        # fig.savefig(f"{area}/{current_sample}.png")
                        plt.close(fig)

                        current_sample += 1

                    return [board for board in boards]

            return []

        paths = []
        if j < area - 1:
            skip = positions.copy()
            skip[i, j+1] = 1
            paths.extend(path_from(skip, i, j + 1, mask))

        if i > 0:
            positions[i-1, j] = 1
            paths.extend(path_from(positions, i - 1, j, mask))

        return paths

    positions = np.zeros((area, area))
    positions[area-1, 0] = 1

    paths = path_from(positions, area - 1, 0, mask)
    print(c)
    return paths


if __name__ == '__main__':
    area = 16
    paths = find_path(area, batch_size=1024)



