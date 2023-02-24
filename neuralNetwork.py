from snakeIO import *


""""
    Input layer has 19 nodes:  HeadX, HeadY, Length,
                            UpToApple, DownToApple, LeftToApple, RightToApple,
                            UpToWall, DownToWall, LeftToWall, RighToWall,
                            UpToTail, DownToTail, LeftToTail, RightToTail
                            TailUp, TailDown, TailLeft, TailRight
    Hidden layer has 15 nodes
    Output layer has 4 nodes: Up, Down, Left, Right
"""


class NeuralNetwork:
    input_layer = []
    hidden_layer = []
    output_layer = []

    def __init__(self, input_layer, hidden_layer):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = [Node([], 0) for _ in range(4)]

    def calculate_move(self):
        temp_list = compute_activations(self.input_layer)
        bias_list = [n.bias for n in self.hidden_layer]
        # print("bias_list is:")
        # print(bias_list)
        activation_list = [a + b for a, b in zip(temp_list, bias_list)]
        # print("activation_list is: ")
        # print(activation_list)
        for h, t in zip(self.hidden_layer, activation_list):
            h.activation = t

        temp_list = compute_activations(self.hidden_layer)
        bias_list = [n.bias for n in self.output_layer]
        # print("bias_list is:")
        # print(bias_list)
        activation_list = [a + b for a, b in zip(temp_list, bias_list)]
        # print("activation_list is: ")
        # print(activation_list)

        max_output = ("U", -1000000)
        for i, a in zip(["U", "D", "L", "R"], activation_list):
            if a > max_output[1]:
                max_output = (i, a)

        return max_output[0]

        # for o, t in zip(self.output_layer, activation_list):
        #     o.activation = t

    def set_activation(self, activations):
        for i, a in zip(self.input_layer, activations):
            i.activation = a


if __name__ == "__main__":
    print("Ran from neuralNetwork.py")

    layer1, layer2 = read_agent("agent1")
    randomize_agent(layer1, layer2)

    nn1 = NeuralNetwork(layer1, layer2)
    nn1.set_activation([rng.random() for _ in range(19)])

    nn1.calculate_move()

    print("Output layer is:")
    for n in nn1.output_layer:
        print(n.activation)