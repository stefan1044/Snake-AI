import numpy as np
import time as time

rng = np.random.default_rng(int(time.time()))


class Node:
    weights = []
    bias = 0
    activation = 0

    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias

    def init_random(self) -> None:
        self.weights = [rng.random() * (-1 if rng.random() < 0.5 else 1)
                        for _ in range(len(self.weights))]
        self.bias = int(rng.random() * 10 * (-1 if rng.random() < 0.5 else 1))


def relu(activation):
    return max(0.0, activation)


def compute_activations(layer):

    weight_matrix = np.array([n.weights for n in layer]).transpose()
    activation_matrix = np.array([[n.activation for n in layer], ]).transpose()

    # print("Weight matrix is:")
    # print(weight_matrix)
    # print("Activation matrix is:")
    # print(activation_matrix)
    # print("--------------------")

    activation = [a.item() for a in np.matmul(weight_matrix, activation_matrix)]
    # print("Activation list is:")
    # print(activation)

    return activation


if __name__ == "__main__":
    print(rng.random())
