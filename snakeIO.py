from neurons import *


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def read_agent(name: str) -> (list[Node], list[Node]):
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


def read_agent_layers(name: str) -> (list[float], list[float]):
    r = open("agents/" + name + ".txt", "r")

    float_list = []
    for line in r.readlines():
        float_list.append(float(line))

    r.close()

    input_layer = float_list[:304]
    hidden_layer = float_list[304:]

    return input_layer, hidden_layer


def update_agent(name: str, input_layer: list[Node], hidden_layer: list[Node]):
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


def copy_over_agents():
    for i in range(51):
        n1, n2 = read_agent("genetic" + str(i) + "agent")
        update_agent("agent" + str(i), n1, n2)


if __name__ == "__main__":
    print("Ran from snakeIO.py")

    copy_over_agents()

