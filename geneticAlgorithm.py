from gameLogic import *
from snakeIO import *
from copy import deepcopy

crossover_chance = 0
mutation_chance = 0
number_elites = 0
tournament_size = 0
number_generations = 0
board_size = 0


def run_algorithm(nodes: list[list]):

    agent_number = len(nodes)

    current_generation = 0
    while current_generation < number_generations:
        neural_networks = [NeuralNetwork(layer[0], layer[1]) for layer in nodes]
        agents = [nn.to_list() for nn in neural_networks]

        evaluate = [run_game([5], nn) for nn in neural_networks]
        fitness = [(snake_length - snake_moves / board_size, i) for e, i in zip(evaluate, range(agent_number)) \
                   for snake_length, snake_moves in zip(e[0], e[1])]

        r = open("agents/debugSize" + str(board_size) + ".txt", "w")
        [r.write(e[2]) for e in evaluate]

        tournament_winners = [max(fitness[j][0] for j in t) for t in
                              [rng.integers(0, agent_number, tournament_size) for i in
                               range(agent_number - number_elites)]]

        children_agents = [crossover(agents[i], agents[i + 1]) for i in range(len(tournament_winners) - 1, step=2)]
        fitness.sort(reverse=True)
        for i in range(number_elites):
            children_agents.append(agents[fitness[i][1]])
        agents = children_agents.copy()

    for i in range(len(agents)):
        update_agent("genetic" + str(i) + "agent", agents[i][0], agents[i][1])


def crossover(agent1: tuple, agent2: tuple):
    if crossover_chance < rng.random():
        return agent1, agent2

    child1 = deepcopy(agent1)
    child2 = deepcopy(agent2)
    # cross-over input layer
    crossover_point = rng.integers(1, 303)
    for i in range(crossover_point, 304):
        child1[0][i], child2[0][i] = child2[0][i], child1[0][i]
        if rng.random() < mutation_chance:
            child1[0][i] += rng.random()*child1[0][i]
        if rng.random() < mutation_chance:
            child2[0][i] += rng.random()*child2[0][i]

    # cross-over hidden layer
    crossover_point = rng.integers(1, 74)
    for i in range(crossover_point, 75):
        child1[1][i], child2[1][i] = child2[1][i], child1[1][i]
        if rng.random() < mutation_chance:
            child1[1][i] += rng.random()*child1[1][i]
        if rng.random() < mutation_chance:
            child2[1][i] += rng.random()*child2[1][i]

    return child1, child2


if __name__ == "__main__":
    print("Ran from geneticAlgorithm.py")

    crossover_chance = 0.8
    mutation_chance = 0.01
    number_elites = 5
    tournament_size = 3
    number_generations = 3
    board_size = 5

    agent_layers = [read_agent("agent" + str(i)) for i in range(51)]
    # neural_networks = [NeuralNetwork(layer[0], layer[1]) for layer in agents]
    # print([nn.to_list() for nn in neural_networks])

    run_algorithm(agent_layers)
