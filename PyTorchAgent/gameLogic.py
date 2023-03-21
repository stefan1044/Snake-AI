import itertools
from enum import Enum
from collections import deque
import numpy as np
import pygame

pygame.init()
screen = pygame.display.set_mode((1280, 720))


class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class SnakeGame:
    old_head: tuple[int, int] = None
    snake_length: int = None
    maximum_moves: int = None
    moves_since_apple: int = None
    moves_total: int = None
    snake_positions: deque = None
    snake_head: tuple[int, int] = None
    free_positions: list[tuple[int, int]] = None
    apple: tuple[int, int] = None
    last_move: Direction = None
    board_size: int = None

    def __init__(self, board_size: int):
        # render_board(screen, board_size)
        self.board_size = board_size
        self.reset()

    def new_free_position(self) -> tuple[int, int]:
        return self.free_positions[np.random.randint(0, len(self.free_positions))]

    def create_apple(self):
        try:
            self.apple = self.new_free_position()
        except KeyError:  # finished all apples
            print("Had Key Error")

    def move_snake(self, action: list[int]):
        # calculate the direction based on the action taken
        clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = clock_wise.index(self.last_move)

        if np.array_equal(action, [0, 1, 0]):
            direction = clock_wise[idx]
        elif np.array_equal(action, [1, 0, 0]):
            next_idx = (idx - 1) % 4
            direction = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (idx + 1) % 4
            direction = clock_wise[next_idx]

        # print(f"Action is {action}, last move is {self.last_move}, direction is {direction}")

        self.last_move = direction
        if direction == Direction.UP:
            direction_vector = (1, 0)
        elif direction == Direction.DOWN:
            direction_vector = (-1, 0)
        elif direction == Direction.LEFT:
            direction_vector = (0, -1)
        else:
            direction_vector = (0, 1)

        self.old_head = self.snake_head
        self.snake_head = self.snake_head[0] + direction_vector[0], self.snake_head[1] + direction_vector[1]

    def check_wall_collision(self) -> bool:
        if self.snake_head[0] < 1 or self.snake_head[0] > self.board_size:
            return True
        if self.snake_head[1] < 1 or self.snake_head[1] > self.board_size:
            return True
        return False

    def check_tail_collision(self) -> bool:
        # if len(self.snake_positions) < 2:
        #     return False
        for position in deque(itertools.islice(self.snake_positions, 0, len(self.snake_positions) - 1)):
            if self.snake_head == position:
                return True
        return False

    def reset(self):
        # generate starting position
        temp_position_list = [i for i in range(1, self.board_size + 1)]
        self.free_positions = list(itertools.product(temp_position_list, temp_position_list))
        self.snake_positions = deque()
        self.snake_head = self.new_free_position()
        self.free_positions.remove(self.snake_head)

        if self.snake_head[0] > 1:
            self.last_move = Direction.UP
            self.snake_positions.append((self.snake_head[0] - 1, self.snake_head[1]))
            self.free_positions.remove((self.snake_head[0] - 1, self.snake_head[1]))
        else:
            self.last_move = Direction.DOWN
            self.snake_positions.append((self.snake_head[0] + 1, self.snake_head[1]))
            self.free_positions.remove((self.snake_head[0] + 1, self.snake_head[1]))

        self.snake_length = 2

        # generate first apple
        self.create_apple()

        self.moves_total = 0
        self.moves_since_apple = 0
        self.maximum_moves = self.board_size ** 2

    def play_step(self, action: list[int]) -> tuple[int, int, int, bool]:

        self.move_snake(action)
        self.snake_positions.appendleft(self.old_head)
        self.moves_total += 1
        self.moves_since_apple += 1

        if self.check_wall_collision() or self.check_tail_collision():
            self.snake_positions.pop()
            return self.snake_length, self.moves_total, -10, True

        if self.moves_since_apple > self.maximum_moves:
            return self.snake_length, self.moves_total, -15, True

        if self.snake_head == self.apple:
            self.snake_length += 1
            self.create_apple()
            self.moves_since_apple = 0
            return self.snake_length, self.moves_total, 30, False

        # moved to an empty tile
        self.free_positions.append(self.snake_positions.pop())
        # print(f"Tried popping {self.snake_head}, move was {self.last_move}")
        self.free_positions.remove(self.snake_head)
        return self.snake_length, self.moves_total, 0, False


def render_board(surface: pygame.Surface, game: SnakeGame):
    square_size = int(700 / game.board_size)

    surface.fill((0, 0, 0))
    color = 0
    width = 0

    # draw board
    for pos in game.free_positions:
        pygame.draw.rect(surface, (50, 50, 50), pygame.Rect((pos[1]-1) * square_size,
                                                            (pos[0]-1) * square_size, square_size, square_size),
                         2)

    # draw head
    pygame.draw.rect(surface, (0, 0, 255), pygame.Rect((game.snake_head[1]-1) * square_size,
                                                       (game.snake_head[0]-1) * square_size, square_size, square_size),
                     width)

    # draw apple
    pygame.draw.rect(surface, (255, 0, 0), pygame.Rect((game.apple[1]-1) * square_size,
                                                       (game.apple[0]-1) * square_size, square_size, square_size),
                     width)

    # draw snake
    for pos in game.snake_positions:
        pygame.draw.rect(surface, (0, 255, 0), pygame.Rect((pos[1]-1) * square_size,
                                                           (pos[0]-1) * square_size, square_size, square_size),
                         width)


if __name__ == "__main__":
    print("Ran from gameLogic.py")
