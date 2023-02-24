import itertools
import pygame

from gameLogic import translate_direction
from neuralNetwork import *
from collections import deque


def render_board(surface, board):
    board_size = board.shape[0] + 2

    square_size = int(700/board_size)

    surface.fill((0, 0, 0))
    it = np.nditer(board, flags=['multi_index'])
    color = 0
    width = 0
    for x in it:
        if x == 3:
            color = (255, 255, 255)
        elif x == 2:
            color = (255, 0, 0)
        elif x == 1:
            color = (0, 255, 0)
        elif x == 0:
            color = (255, 255, 255)
            width = 2
        pygame.draw.rect(surface, color, pygame.Rect(350 + it.multi_index[1] * square_size,
                                                     50 + it.multi_index[0] * square_size, square_size, square_size),
                         width)
        width = 0
    pass


def render_game(game_string: list):
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))

    move_string = game_string[2]
    board_size = move_string[0]
    move_string.pop(0)
    start_pos = move_string[0]
    move_string.pop(0)
    apple_pos = move_string[0]
    move_string.pop(0)

    board = np.zeros((board_size + 2, board_size + 2), dtype=np.int8)
    # border
    board[0] = [3 for _ in range(board_size + 2)]
    board[board_size + 1] = [3 for _ in range(board_size + 2)]
    board[:, 0] = [3 for _ in range(board_size + 2)]
    board[:, board_size + 1] = [3 for _ in range(board_size + 2)]

    board[start_pos[0], start_pos[1]] = 1
    board[apple_pos[0], apple_pos[1]] = 2

    snake_positions = deque()
    snake_positions.append(start_pos)

    for move in move_string:
        render_board(screen, board)
        pygame.display.flip()
        if move[0] == "A":
            direction = translate_direction(move[3])
            board[move[0], move[1]] = 2
            board[snake_positions[0][0] + direction[0], snake_positions[0][1] + direction[1]] = 1
            continue

        direction = translate_direction(move)
        snake_positions.appendleft((snake_positions[0][0] + direction[0], snake_positions[0][1] + direction[1]))
        board[snake_positions[0][0], snake_positions[0][1]] = 1
        taily, tailx = snake_positions.pop()
        board[taily, tailx] = 0
        time.sleep(0.1)

    render_board(screen, board)
    pygame.display.flip()
    time.sleep(1)
    pygame.quit()


if __name__ == "__main__":
    print("Ran from gameRender.py")

    render_game()
