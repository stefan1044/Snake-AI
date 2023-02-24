
from gameRender import *
from gameLogic import *

if __name__ == "__main__":
    print("Ran from main.py")

    layer1, layer2 = read_agent("agent1")
    nn1 = NeuralNetwork(layer1, layer2)

    render_game(run_game([5], nn1))

