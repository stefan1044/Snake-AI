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
    input_layer: list[Node] = []
    hidden_layer: list[Node] = []
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
        # print(f"activation list is:\n{activation_list}")

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

    def randomize_network(self):
        self.input_layer = [Node([0 for _ in range(15)], 0) for _ in range(19)]
        self.hidden_layer = [Node([0 for _ in range(4)], 0) for _ in range(15)]
        for node in self.input_layer:
            node.init_random()
        for node in self.hidden_layer:
            node.init_random()

    def to_list(self) -> list:
        value_list = []
        for n in self.input_layer:
            for v in n.weights:
                value_list.append(v)
            value_list.append(n.bias)
        for n in self.hidden_layer:
            for v in n.weights:
                value_list.append(v)
            value_list.append(n.bias)

        return value_list


def to_layers(values: list[list]):
    return_values = []

    for value in values:
        input_layer = []
        for group in chunker(value[:304], 16):
            input_layer.append(Node(group[:15], group[15]))

        hidden_layer = []
        for group in chunker(value[304:], 5):
            hidden_layer.append(Node(group[:4], group[4]))

        return_values.append((input_layer, hidden_layer))

    return return_values


if __name__ == "__main__":
    print("Ran from neuralNetwork.py")

    layer1, layer2 = read_agent("agent1")