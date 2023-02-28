import itertools

from neuralNetwork import *
from collections import deque


def translate_direction(direction):
    if direction == "U":
        return 1, 0
    elif direction == "D":
        return -1, 0
    elif direction == "L":
        return 0, -1
    elif direction == "R":
        return 0, 1
    else:
        raise Exception("Direction " + direction + " is not a valid direction!")
    pass


def pick_random(free_positions: list):
    return free_positions[np.random.randint(0, len(free_positions))]


def run_game(game_settings, agent: NeuralNetwork):
    board_size = game_settings[0]
    board = np.zeros((board_size + 2, board_size + 2), dtype=np.int8)

    # border
    board[0] = [3 for _ in range(board_size + 2)]
    board[board_size + 1] = [3 for _ in range(board_size + 2)]
    board[:, 0] = [3 for _ in range(board_size + 2)]
    board[:, board_size + 1] = [3 for _ in range(board_size + 2)]

    # generate starting position
    temp_position_list = [i for i in range(1, board_size + 1)]
    free_positions = list(itertools.product(temp_position_list, temp_position_list))
    snake_positions = deque()
    snake_positions.append(pick_random(free_positions))
    # head of the snake will be at snake_positions[0], y being snake_positions[0][0] and x being snake_positions[0][1]
    last_move = 0
    board[snake_positions[0][0], snake_positions[0][1]] = 1
    snake_length = 1

    # generate first apple
    appley, applex = pick_random(free_positions)
    board[appley, applex] = 2

    # print(f"Starting board:\n{board}")

    moves_total = 1
    moves_since_apple = 0
    maximum_moves = board_size * 10
    while moves_since_apple < maximum_moves:
        # print(f"moves_since_apple :{moves_since_apple}, length:{snake_length}")
        apple_up = snake_positions[0][0] - appley
        apple_left = snake_positions[0][1] - applex

        # distance to tail in each direction
        up_tail, down_tail, left_tail, right_tail = -1, -1, -1, -1
        for y in range(snake_positions[0][0] - 1, 0, -1):
            if board[y, snake_positions[0][1]] == 1:
                up_tail = snake_positions[0][0] - y
                break
        for y in range(snake_positions[0][0] + 1, board_size + 1):
            if board[y, snake_positions[0][1]] == 1:
                down_tail = y - snake_positions[0][0]
                break
        for x in range(snake_positions[0][1] - 1, 0, -1):
            if board[snake_positions[0][0], x] == 1:
                left_tail = snake_positions[0][1] - x
                break
        for x in range(snake_positions[0][1] + 1, board_size + 1):
            if board[snake_positions[0][0], x] == 1:
                right_tail = x - snake_positions[0][1]
                break

        # activation layer
        activation = [
            snake_positions[0][1], snake_positions[0][0], snake_length,
            apple_up, -apple_up, apple_left, -apple_left,
            snake_positions[0][0], board_size - snake_positions[0][0] + 1, snake_positions[0][1],
            board_size - snake_positions[0][1] + 1,
            up_tail, down_tail, left_tail, right_tail,
        ]
        if last_move == "U":
            activation.extend([1, 0, 0, 0])
        elif last_move == "D":
            activation.extend([0, 1, 0, 0])
        elif last_move == "L":
            activation.extend([0, 0, 1, 0])
        elif last_move == "R":
            activation.extend([0, 0, 0, 1])
        else:
            activation.extend([0, 0, 0, 0])

        agent.set_activation(activation)
        move = agent.calculate_move()

        direction = translate_direction(move)
        nextx = snake_positions[0][1] + direction[1]
        nexty = snake_positions[0][0] + direction[0]
        new_tile = board[nexty, nextx]
        board[nexty, nextx] = 1

        # print(f"Board at move {moves_total} is:\n{board}")

        if new_tile == 0:  # tile is empty
            snake_positions.appendleft((nexty, nextx))
            taily, tailx = snake_positions.pop()
            board[taily, tailx] = 0
        elif new_tile == 1:  # bit tail
            return snake_length
        elif new_tile == 2:  # ate apple
            snake_positions.appendleft((nexty, nextx))
            snake_length += 1
            # generate next apple
            try:
                appley, applex = pick_random(free_positions)
            except KeyError:  # finished all apples
                print("Had Key Error")
                return snake_length
            board[appley, applex] = 2
            moves_since_apple = -1
        elif new_tile == 3:  # hit wall
            return snake_length

        moves_since_apple += 1
        moves_total += 1

    return snake_length


def run_game_debug(game_settings, agent: NeuralNetwork):
    board_size = game_settings[0]
    board = np.zeros((board_size + 2, board_size + 2), dtype=np.int8)
    move_string = [board_size]

    # border
    board[0] = [3 for _ in range(board_size + 2)]
    board[board_size + 1] = [3 for _ in range(board_size + 2)]
    board[:, 0] = [3 for _ in range(board_size + 2)]
    board[:, board_size + 1] = [3 for _ in range(board_size + 2)]

    # generate starting position
    temp_position_list = [i for i in range(1, board_size + 1)]
    free_positions = list(itertools.product(temp_position_list, temp_position_list))
    snake_positions = deque()
    snake_positions.append(pick_random(free_positions))
    # head of the snake will be at snake_positions[0], y being snake_positions[0][0] and x being snake_positions[0][1]
    last_move = 0
    board[snake_positions[0][0], snake_positions[0][1]] = 1
    move_string.append((snake_positions[0][0], snake_positions[0][1]))
    snake_length = 1

    # generate first apple
    appley, applex = pick_random(free_positions)
    board[appley, applex] = 2
    move_string.append((appley, applex))

    # print(f"Starting board:\n{board}")

    moves_total = 1
    moves_since_apple = 0
    maximum_moves = board_size * 10
    while moves_since_apple < maximum_moves:
        # print(f"moves_since_apple :{moves_since_apple}, length:{snake_length}")
        apple_up = snake_positions[0][0] - appley
        apple_left = snake_positions[0][1] - applex

        # distance to tail in each direction
        up_tail, down_tail, left_tail, right_tail = -1, -1, -1, -1
        for y in range(snake_positions[0][0] - 1, 0, -1):
            if board[y, snake_positions[0][1]] == 1:
                up_tail = snake_positions[0][0] - y
                break
        for y in range(snake_positions[0][0] + 1, board_size + 1):
            if board[y, snake_positions[0][1]] == 1:
                down_tail = y - snake_positions[0][0]
                break
        for x in range(snake_positions[0][1] - 1, 0, -1):
            if board[snake_positions[0][0], x] == 1:
                left_tail = snake_positions[0][1] - x
                break
        for x in range(snake_positions[0][1] + 1, board_size + 1):
            if board[snake_positions[0][0], x] == 1:
                right_tail = x - snake_positions[0][1]
                break

        # activation layer
        activation = [
            snake_positions[0][1], snake_positions[0][0], snake_length,
            apple_up, -apple_up, apple_left, -apple_left,
            snake_positions[0][0], board_size - snake_positions[0][0] + 1, snake_positions[0][1],
            board_size - snake_positions[0][1] + 1,
            up_tail, down_tail, left_tail, right_tail,
        ]
        if last_move == "U":
            activation.extend([1, 0, 0, 0])
        elif last_move == "D":
            activation.extend([0, 1, 0, 0])
        elif last_move == "L":
            activation.extend([0, 0, 1, 0])
        elif last_move == "R":
            activation.extend([0, 0, 0, 1])
        else:
            activation.extend([0, 0, 0, 0])

        agent.set_activation(activation)
        move = agent.calculate_move()

        direction = translate_direction(move)
        nextx = snake_positions[0][1] + direction[1]
        nexty = snake_positions[0][0] + direction[0]
        new_tile = board[nexty, nextx]
        board[nexty, nextx] = 1

        # print(f"Board at move {moves_total} is:\n{board}")

        if new_tile == 0:  # tile is empty
            snake_positions.appendleft((nexty, nextx))
            taily, tailx = snake_positions.pop()
            board[taily, tailx] = 0
            move_string.append(move)
        elif new_tile == 1:  # bit tail
            move_string.append(move)
            return snake_length, moves_total, move_string
        elif new_tile == 2:  # ate apple
            snake_positions.appendleft((nexty, nextx))
            snake_length += 1
            # generate next apple
            try:
                appley, applex = pick_random(free_positions)
            except KeyError:  # finished all apples
                print("Had Key Error")
                return snake_length, moves_total, move_string
            move_string.append(("A", appley, applex, move))
            board[appley, applex] = 2
            moves_since_apple = -1
        elif new_tile == 3:  # hit wall
            move_string.append(move)
            return snake_length, moves_total, move_string

        moves_since_apple += 1
        moves_total += 1

    return snake_length, moves_total, move_string

if __name__ == "__main__":
    print("Ran from gameLogic.py")

    layer1, layer2 = read_agent("genetic1agent")

    nn1 = NeuralNetwork(layer1, layer2)

    game_string = run_game([5], nn1)
    print(f"Game string is {game_string}")
    print("Ended")
