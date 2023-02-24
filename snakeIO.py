from neurons import *


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def read_agent(name):
    r = open("agents/" + name + ".txt", "r")

    float_list = []
    for line in r.readlines():
        float_list.append(float(line))

    r.close()

    input_layer = []
    for group in chunker(float_list[:304], 16):
        input_layer.append(Node(group[:15], group[15]))

    hidden_layer = []
    for group in chunker(float_list[304:], 5):
        hidden_layer.append(Node(group[:4], group[4]))

    return input_layer, hidden_layer


def randomize_agent(input_layer, hidden_layer):
    for n in input_layer:
        n.init_random()
    for n in hidden_layer:
        n.init_random()


if __name__ == "__main__":
    print("Ran from snakeIO.py")

    test1, test2 = read_agent("agent1")

    randomize_agent(test1, test2)

    for n in test1:
        print(str(n.weights) + str(n.bias))

    for n in test2:
        print(str(n.weights) + str(n.bias))

