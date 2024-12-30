"""
Script for visualizing effect of visited cells. It uses pygame to create an interactive environment to visit and "unvisit"
each cell and see how it affects the filled world cells base on priority map.
"""

import pygame
import numpy as np
import matplotlib.cm as cm
import argparse
from utils import priority_map, does_fill

pygame.init()

def parse_args():
    parser = argparse.ArgumentParser(description="Priority map visualization")
    parser.add_argument("--area", type=int, default=36, help="Area size (default: 36)")
    parser.add_argument("--world_size", type=int, default=36, help="World/map size (default: 36)")
    parser.add_argument("--cell_size", type=int, default=25, help="Cell size in pixels (default: 25)")
    return parser.parse_args()

def draw_grid(screen, board, map, cell_size, array_size):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    screen.fill(WHITE)
    font = pygame.font.SysFont(None, 25)

    colormap = cm.get_cmap('viridis')
    for row in range(array_size[0]):
        for col in range(array_size[1]):
            normalized_value = map[row, col] / (np.max(map) + 0.0001)
            value = map[row][col]
            color = colormap(normalized_value)
            pygame_color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

            if board[row][col] or normalized_value == -10:
                pygame_color = (0, 0, 0)

            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, pygame_color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            text_surface = font.render(f"{value:.0f}", True, WHITE if value > 0.5 else BLACK)
            text_rect = text_surface.get_rect(center=rect.center)
            screen.blit(text_surface, text_rect)

def handle_click(pos, board, map, area, map_size, cell_size, array_size):
    x, y = pos
    col = x // cell_size
    row = y // cell_size

    if 0 <= row < array_size[0] and 0 <= col < array_size[1]:
        board[row, col] = ~board[row, col]
        map[:] = priority_map(board, area, map_size=map_size)

def main():
    args = parse_args()

    area = args.area
    map_size = args.world_size
    cell_size = args.cell_size
    array_size = (map_size, map_size)

    board = np.zeros(array_size, dtype=bool)

    board[0, 0] = True


    map = priority_map(board, area, map_size=map_size)

    screen_width = array_size[1] * cell_size
    screen_height = array_size[0] * cell_size

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Priority map")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(event.pos, board, map, area, map_size, cell_size, array_size)
                print(does_fill(board, area, symmetric=False, return_missed=True))
                print(board.sum())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    board.fill(False)
                    map[:] = priority_map(board, area, map_size=map_size)

        draw_grid(screen, board, map, cell_size, array_size)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
