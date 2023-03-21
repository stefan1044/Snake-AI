import random
import time

from gameLogic import *
from model import *
from helper import *


MAX_MEMORY = 10_000
BATCH_SIZE = 1_000
LR = 0.001


class Agent:
    games_ran: int = None
    epsilon: int = None
    gamma: float = None
    memory: deque = None
    model: LinerQNet = None
    trainer: QTrainer = None

    def __init__(self, file_name):
        self.games_ran = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        if file_name != "":
            self.model = torch.load(file_name)
        else:
            self.model = LinerQNet(12, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    @staticmethod
    def get_input(game_object: SnakeGame) -> np.ndarray:
        # headx, heady,
        # direction_up, direction_right, direction_down, direction_left (one hot-encoded)
        # danger_up, danger_right, danger_down, danger_left
        # appley, applex

        heady, headx = game_object.snake_head

        direction_up = game_object.last_move == Direction.UP
        direction_right = game_object.last_move == Direction.RIGHT
        direction_down = game_object.last_move == Direction.DOWN
        direction_left = game_object.last_move == Direction.LEFT

        danger_up = 0
        for y in range(heady + 1, game_object.board_size + 1):
            for pos in game_object.snake_positions:
                if (y, headx) == pos:
                    danger_up = y - heady
                    break
            else:
                continue
            break
        if danger_up == 0:
            danger_up = game_object.board_size - heady + 1

        danger_right = 0
        for x in range(headx + 1, game_object.board_size + 1):
            for pos in game_object.snake_positions:
                if (heady, x) == pos:
                    danger_right = x - headx
                    break
            else:
                continue
            break
        if danger_right == 0:
            danger_right = game_object.board_size - headx + 1

        danger_down = 0
        for y in range(heady - 1, 0, -1):
            for pos in game_object.snake_positions:
                if (y, headx) == pos:
                    danger_down = heady - y
                    break
            else:
                continue
            break
        if danger_down == 0:
            danger_down = heady

        danger_left = 0
        for x in range(headx - 1, 0, -1):
            for pos in game_object.snake_positions:
                if (heady, x) == pos:
                    danger_left = headx - x
                    break
            else:
                continue
            break
        if danger_left == 0:
            danger_left = headx

        appley, applex = game_object.apple

        game_input = [heady, headx,
                      direction_up, direction_right, direction_down, direction_left,
                      danger_up, danger_right, danger_down, danger_left,
                      appley, applex]

        # print(f"Game input is {game_input}, length is {game_object.snake_length}\n\n")

        return np.array(game_input, dtype=int)

    def remember(self, game_input: np.ndarray, action: list[int], reward: int, next_state, done: bool):
        self.memory.append((game_input, action, reward, next_state, done))

    def train_short_memory(self, game_input: np.ndarray, action: list[int], reward: int, next_state, done: bool):
        self.trainer.train_step(game_input, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        game_inputs, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(game_inputs, actions, rewards, next_states, dones)

    def determine_move(self, input_information: np.ndarray) -> list[int]:

        move = [0, 0, 0]

        tensor = torch.tensor(input_information, dtype=torch.float)
        prediction = self.model(tensor)
        move[torch.argmax(prediction).item()] = 1

        return move


def start_training(board_size: int, file_name=""):
    scores = []
    average_scores = []
    total_score = 0
    record = 0
    agent = Agent(file_name)
    print(f"Debug values: {agent.games_ran}")
    game = SnakeGame(board_size)

    while True:
        if agent.games_ran > 5000:
            render_board(screen, game)
            pygame.display.flip()
            time.sleep(0.1)
        input_information = agent.get_input(game)
        # print(f"Debug values:\nsnake_head: {game.snake_head}\napple: {game.apple}\n"
        #       f"snake_positions: {game.snake_positions}\nfree_positions: {game.free_positions}\nInput Information:"
        #       f"{input_information}")

        move = agent.determine_move(input_information)

        snake_length, total_moves, reward, done = game.play_step(move)
        score = snake_length - 2 - total_moves/(board_size**2)
        new_input = agent.get_input(game)

        agent.train_short_memory(input_information, move, reward, new_input, done)
        agent.remember(input_information, move, reward, new_input, done)

        if done:
            game.reset()
            agent.games_ran += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(file_name + "new.pth")
                print(f"New record is: {record}, with model:\n{torch.load(file_name + 'new.pth')}")

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.games_ran
            average_scores.append(mean_score)

            # print(f"Finished game {str(agent.games_ran)}  with score {score}(length: {snake_length},"
            #       f" moves: {total_moves}) . Current record: {record}, average: {mean_score}")
            plot(scores, average_scores)
            # print(average_scores)


if __name__ == "__main__":
    print("Ran from agent.py")

    start_training(5, "new.pth")
