import numpy as np


def wrap_position(pos, max_value):
    if pos < 0:
        return max_value - 1
    elif pos >= max_value:
        return 0
    return pos


class GameState:
    """
    Class for simulating the game. It tracks the changes in snake position and updates the state variable depending
    on whether the algorithm managed to fill the whole world within limit or not.
    """

    def __init__(self, world_width, world_height, M=35):
        self.world_width = world_width
        self.world_height = world_height
        self.M = M
        self.snake_pos = [0, 0]

        self.game_state = np.zeros((world_width, world_height), dtype=int)

        self.world_area = world_width * world_height
        self.moves = 0
        self.filled = 1
        self.state = 0

    def update_state(self, direction):
        self.game_state[self.snake_pos[0], self.snake_pos[1]] = 3

        self.snake_pos[0] += direction[0]
        self.snake_pos[1] += direction[1]

        # Wrap the snake's position around the game world
        self.snake_pos[0] = wrap_position(self.snake_pos[0], self.world_width)
        self.snake_pos[1] = wrap_position(self.snake_pos[1], self.world_height)

        if self.game_state[self.snake_pos[0], self.snake_pos[1]] != 3:
            self.filled += 1

        self.game_state[self.snake_pos[0], self.snake_pos[1]] = 1

        self.moves += 1

        if self.filled == self.world_area:
            self.state = 1
            return

        if self.moves > self.M * self.world_area and self.filled < self.world_area:
            self.state = -1

