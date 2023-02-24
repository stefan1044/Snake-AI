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


def update_agent(name, input_layer, hidden_layer):
    r = open("agents/" + name + ".txt", "w")

    for node in input_layer:
        for weight in node.weights:
            r.write(str(weight) + "\n")
        r.write(str(node.bias) + "\n")

    for node in hidden_layer:
        for weight in node.weights:
            r.write(str(weight) + "\n")
        r.write(str(node.bias) + "\n")

    r.close()


if __name__ == "__main__":
    print("Ran from snakeIO.py")

    test1, test2 = read_agent("agent1")

    for nr in test1:
        print(str(nr.weights) + str(nr.bias))

    for nr in test2:
        print(str(nr.weights) + str(nr.bias))

