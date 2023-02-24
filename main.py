from neurons import *

if __name__ == "__main__":

    node1 = Node([1, 2, 3], 10)
    node1.activation = 6
    node2 = Node([4, 5, 6], 15)
    node2.activation = 15
    node3 = Node([7, 8, 9], 20)
    node3.activation = 24

    node2.init_random()
    node3.init_random()
    node1.init_random()

    nodes = [node1, node2, node3]

    print(compute_activations(nodes))
