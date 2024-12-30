"""
Utils functions for analyzing snake paths.
"""

import torch
import numpy as np


def split_and_or_tiles(array: torch.tensor, tile_height, tile_width):
    """
    Splits a 2d bool array into tiles of equal size and does or operations along tile dimension.
    """

    # Get the shape of the input array
    rows, cols = array.shape

    # Calculate the padding needed to make dimensions divisible by tile size
    pad_rows = (tile_height - rows % tile_height) % tile_height
    pad_cols = (tile_width - cols % tile_width) % tile_width

    # Pad the array with zeros
    padded_array = torch.nn.functional.pad(array, (0, pad_cols, 0, pad_rows), mode='constant', value=0)

    # Get the new shape of the padded array
    padded_rows, padded_cols = padded_array.shape

    # Calculate the number of tiles along each dimension
    num_tiles_row = padded_rows // tile_height
    num_tiles_col = padded_cols // tile_width

    # Reshape the padded array into tiles
    tiles = padded_array.view(num_tiles_row, tile_height, num_tiles_col, tile_width)

    # Permute the axes to group tiles together
    tiles = tiles.permute(0, 2, 1, 3)  # (num_tiles_row, num_tiles_col, tile_height, tile_width)

    # Perform logical OR along the tile dimensions (tile_height, tile_width)
    result = torch.any(tiles, dim=(0, 1))

    return result


def does_fill(array: np.ndarray, area, symmetric=True, return_missed=False, device='cuda'):
    """
    Checks whether given snake path given as 2d bool array of visited cells fills all worlds with area less or equal to
    given area.
    """
    array = torch.tensor(array, device=device, dtype=torch.bool)
    failed = False
    missed = []
    if symmetric:
        a_end = int(area ** 0.5) + 1
    else:
        a_end = area + 1

    for a in range(1, a_end):
        if symmetric:
            b_start = (area//a+1) // 2
        else:
            b_start = 1

        for b in range(b_start, area // a + 1):

            filled = split_and_or_tiles(array, a, b)

            if ~torch.all(filled):
                if a != 1 and b != 1:
                    missed.append((a, b))
                if return_missed:
                    failed = True
                else:
                    return False
    if return_missed:
        return not failed, missed
    return not failed


def split_and_or_tiles_batch(array: torch.tensor, tile_height, tile_width):
    """
    Batch version of split_and_or_tiles.
    """

    # Get the shape of the input array
    batch_size, rows, cols = array.shape

    # Calculate the padding needed to make dimensions divisible by tile size
    pad_rows = (tile_height - rows % tile_height) % tile_height
    pad_cols = (tile_width - cols % tile_width) % tile_width

    # Pad the array with zeros
    padded_array = torch.nn.functional.pad(array, (0, pad_cols, 0, pad_rows), mode='constant', value=0)

    # Get the new shape of the padded array
    _, padded_rows, padded_cols = padded_array.shape

    # Calculate the number of tiles along each dimension
    num_tiles_row = padded_rows // tile_height
    num_tiles_col = padded_cols // tile_width

    # Reshape the padded array into tiles
    tiles = padded_array.view(batch_size, num_tiles_row, tile_height, num_tiles_col, tile_width)

    # Perform logical OR along the tile dimensions (tile_height, tile_width)
    result = torch.any(tiles, dim=(1, 3))  # Reduce over tile height and width

    return result


def does_fill_batch(array: torch.tensor, area, device='cuda'):
    """
    Batch version of does_fill.
    """

    fills = torch.ones(array.shape[0], dtype=torch.bool, device=device)
    for a in range(1, int(area ** 0.5) + 1):
        for b in range((area//a+1) // 2, area // a + 1):

            filled = split_and_or_tiles_batch(array, a, b)
            fills = torch.logical_and(torch.all(filled, dim=(1, 2)), fills)

            if ~torch.any(fills):
                return fills
    return fills


def curve_follow(size, loss, start=(-1, 0)):
    board = np.zeros((size, size), dtype=bool)

    i, j = (size+start[0])%size, (size+start[1])%size
    board[i, j] = True

    while i > 0 and j < size - 1:

        up = (i-1, j)
        right = (i, j+1)

        if loss(up) <= loss(right):
            board[up] = True
            i, j = up
        else:
            board[right] = True
            i, j = right

    if i == 0:
        board[0, j:] = True
    else:
        board[0:i, -1] = True

    return board


def priority_map(board, area, map_size=None):
    """
    Creates a priority map based on snake path given as 2d bool array. Each element of the priority map represents how
    many unfilled cells originating from each possible world with area less than given area would be filled if that cell
    was visited next.
    """
    tensor = torch.tensor(board, device='cuda')
    stacks = []

    if map_size is None:
        map_size = area

    pairs = []
    for x in range(1, area + 1):
        for y in range(1, area // x + 1):
            pairs.append((x, y))

    for pair in pairs:
        x, y = pair

        if x == 1 or y == 1:
            continue

        tile = split_and_or_tiles(tensor, x, y)

        v_repetitions = map_size // x + bool(map_size % x)
        h_repetitions = map_size // y + bool(map_size % y)
        scale_factor = 1

        tiled = torch.tile(~tile, (v_repetitions, h_repetitions))
        stacks.append(tiled[:map_size, :map_size] * scale_factor)

    map = torch.stack(stacks, dim=0)
    map = torch.sum(map, dim=0)
    map = map.cpu().numpy()

    return map





