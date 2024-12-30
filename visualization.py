"""
Script used to visualize pathing algorithms. It uses simple pygame and OpenGL logic to render the game state after each
move. It's possible to specify the size of the world, pathing algorithm, frame rate, number of moves per frame,
number of tiles and tile size to visualize the tiling effect.
There are some random bugs when rendering some world sizes for which I did not have time to fix.
Requires pygame and OpenGL to be installed.
"""

import numpy as np
import pygame
import argparse
from OpenGL.GL import *
from game_state import GameState
from pathing import FillPathing, FillPathingOptimized


def update_texture(array):
    h, w = array.shape[:2]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, array)


def draw_grid(screen_size, grid_size, color=(0, 0, 1.0)):
    glColor3f(*color)
    glLineWidth(1)

    glBegin(GL_LINES)
    for x in range(0, screen_size[0] + 1, grid_size):
        glVertex2f(x, 0)
        glVertex2f(x, screen_size[1])
    for y in range(0, screen_size[1] + 1, grid_size):
        glVertex2f(0, y)
        glVertex2f(screen_size[0], y)
    glEnd()

    glColor3f(1.0, 1.0, 1.0)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="A script for visualization of pathing algorithms."
    )

    # Add required arguments
    parser.add_argument(
        "--world_height", "-wh",
        required=False,
        default=100,
        type=int,
        help="World height for game simulation."
    )
    parser.add_argument(
        "--world_width", "-ww",
        required=False,
        default=100,
        type=int,
        help="World width for game simulation."
    )
    parser.add_argument(
        "--pathing_algorithm", "-pa",
        required=False,
        default="fill_pathing",
        type=str,
        help="Pathing algorithm that you wish to visualize, fill_pathing or fill_pathing_optimized."
    )
    parser.add_argument(
        "--frame_rate", "-fps",
        required=False,
        default=60,
        type=int,
        help="Rendering frame rate."
    )
    parser.add_argument(
        "--steps_per_frame", "-spf",
        required=False,
        default=1,
        type=int,
        help="Number of moves to perform for each frame of the simulation."
    )
    parser.add_argument(
        "--num_tiles_vertical",
        required=False,
        default=1,
        type=int,
        help="Number of tiles to render along vertical axis"
    )
    parser.add_argument(
        "--num_tiles_horizontal",
        required=False,
        default=1,
        type=int,
        help="Number of tiles to render along horizontal axis"
    )
    parser.add_argument(
        "--cell_size",
        required=False,
        default=10,
        type=int,
        help="Size of cells in pixels. Overall window size is determined by multiplying world size, number of tiles and cell size."
    )

    args = parser.parse_args()

    num_tiles = (args.num_tiles_horizontal, args.num_tiles_vertical)
    game_size = (args.world_width, args.world_height)
    grid_size = args.cell_size
    tile_size = (game_size[0] * grid_size, game_size[1] * grid_size)
    screen_size = (num_tiles[0] * tile_size[0], num_tiles[1] * tile_size[1])
    fps = args.frame_rate

    steps_per_frame = args.steps_per_frame

    gs = GameState(*game_size)

    if args.pathing_algorithm == "fill_pathing":
        path = FillPathing(35, 1000000)
    else:
        path = FillPathingOptimized(35, 1000000)

    pygame.init()
    pygame.display.set_mode(screen_size, pygame.OPENGL | pygame.DOUBLEBUF)
    glOrtho(0, screen_size[0], 0, screen_size[1], -1, 1)

    color_map = np.array([
        [0, 0, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 0]
    ], dtype=np.uint8)

    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for _ in range(steps_per_frame):
            direction = path.get_direction()
            if not direction:
                print("No more moves.")
                running = False
                break
            gs.update_state(direction)
            if gs.state == 1:
                print("WIN")
                running = False
                break
            if gs.state == -1:
                print("LOSE")
                running = False
                break

        if gs.snake_pos == [0, 0]:  # Avoids some random bug that happens when rendering the frame while snake is at 0,0
            continue

        array = color_map[gs.game_state.T]
        glClear(GL_COLOR_BUFFER_BIT)
        update_texture(array)

        tile_width = tile_size[0]
        tile_height = tile_size[1]
        for row in range(num_tiles[1]):
            for col in range(num_tiles[0]):
                x0 = col * tile_width
                y0 = row * tile_height
                x1 = x0 + tile_width
                y1 = y0 + tile_height

                glBegin(GL_QUADS)
                glTexCoord2f(0, 0)
                glVertex2f(x0, y0)
                glTexCoord2f(1, 0)
                glVertex2f(x1, y0)
                glTexCoord2f(1, 1)
                glVertex2f(x1, y1)
                glTexCoord2f(0, 1)
                glVertex2f(x0, y1)
                glEnd()

        draw_grid(screen_size, grid_size)

        pygame.display.flip()

        clock.tick(fps)

    pygame.quit()


if __name__ == '__main__':
    main()
