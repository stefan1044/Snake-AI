import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


class LinerQNet(nn.Module):
    input_size: int = None
    hidden_size: int = None
    output_size: int = None

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LinerQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x):
        x = f.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        torch.save(self, file_name)

    @staticmethod
    def load(model, file_name=""):
        if file_name == "":
            return
        print(f"Model:\n{torch.load(file_name)}\n\n")
        model = LinerQNet(model.input_size, model.hidden_size, model.output_size)
        model = torch.load(file_name)


class QTrainer:
    def __init__(self, model: LinerQNet, lr: float, gamma: float):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr,)
        self.criterion = nn.MSELoss()

    def train_step(self, game_input: np.ndarray, action: list[int], reward: int, next_game_input: np.ndarray,
                   done: int):
        game_input = torch.tensor(game_input, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_game_input = torch.tensor(next_game_input, dtype=torch.float)

        if len(game_input.shape) == 1:
            game_input = torch.unsqueeze(game_input, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_game_input = torch.unsqueeze(next_game_input, 0)
            done = (done, )

        # predicted values with Q state
        pred = self.model(game_input)

        target = pred.clone()
        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new = reward[i] + self.gamma * torch.max(self.model(next_game_input[i]))

            target[i][torch.argmax(action[i]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


if __name__ == "__main__":
    print("Ran from model.py")
