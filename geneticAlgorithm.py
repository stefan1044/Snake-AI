from gameLogic import *
from snakeIO import *
from copy import deepcopy

crossover_chance = 0
mutation_chance = 0
number_elites = 0
tournament_size = 0
number_generations = 0
board_size = 0
games_to_run = 0


def run_algorithm(nodes: list[list[Node]]):
    agent_number = len(nodes)
    best_generation = []
    best_fitness = 0

    current_generation = 0
    while current_generation < number_generations:
        average_fitness = 0
        print("Started generation " + str(current_generation))
        neural_networks = [NeuralNetwork(layer[0], layer[1]) for layer in nodes]
        agents = [nn.to_list() for nn in neural_networks]

        game_results = [[run_game([board_size], nn) for _ in range(games_to_run)] for nn in neural_networks]
        # e[0] is snake length and e[1] is the number of moves
        fitness = [(sum(g) / games_to_run, i) for g, i in zip(game_results, range(agent_number))]
        s = [i[0] for i in fitness]
        average_fitness = sum(s)/51
        if average_fitness > best_fitness:
            best_fitness = average_fitness
            best_generation = nodes
        print(f"Average fitness of generation is {average_fitness}")

        # r = open("agents/debugSize" + str(board_size) + ".txt", "w")
        # for result in game_results:
        #     for move in result[2]:
        #         r.write(str(move) + ", ")
        #     r.write("\n")

        tournament_winner_values = [max(fitness[j] for j in t) for t in
                                    [rng.integers(0, agent_number, tournament_size) for _ in
                                     range(agent_number - number_elites)]]
        tournament_winners = []
        for w in tournament_winner_values:
            tournament_winners.append(w[1])

        children_agents = []
        for i in range(0, len(tournament_winners), 2):
            c1, c2 = crossover(agents[tournament_winners[i]], agents[tournament_winners[i + 1]])
            children_agents.append(c1)
            children_agents.append(c2)

        fitness.sort(reverse=True)
        for i in range(number_elites):
            children_agents.append(agents[fitness[i][1]])
        nodes = to_layers(children_agents)

        current_generation += 1

    print(f"Best generation fitness is:{best_fitness}")
    for node, i in zip(best_generation, range(0, len(nodes))):
        update_agent("genetic" + str(i) + "agent", node[0], node[1])


def crossover(agent1: list, agent2: list):
    if crossover_chance < rng.random():
        return agent1, agent2

    child1 = deepcopy(agent1)
    child2 = deepcopy(agent2)

    # cross-over input layer
    crossover_point = rng.integers(0, 304)
    for i in range(crossover_point, 304):
        child1[i], child2[i] = child2[i], child1[i]
        if rng.random() < mutation_chance:
            if rng.random() < 0.5:
                child1[i] += (1 - abs(child1[i])) * rng.random() * 0.5
            else:
                child1[i] -= (1 - abs(child1[i])) * rng.random() * 0.5

        if rng.random() < mutation_chance:
            if rng.random() < 0.5:
                child2[i] += (1 - abs(child2[i])) * rng.random() * 0.5
            else:
                child2[i] -= (1 - abs(child1[i])) * rng.random() * 0.5

    # cross-over hidden layer
    crossover_point = rng.integers(304, 379)
    for i in range(crossover_point, 379):
        child1[i], child2[i] = child2[i], child1[i]
        if rng.random() < mutation_chance:
            if rng.random() < 0.5:
                child1[i] += (1 - abs(child1[i])) * rng.random() * 0.5
            else:
                child1[i] -= (1 - abs(child1[i])) * rng.random() * 0.5

        if rng.random() < mutation_chance:
            if rng.random() < 0.5:
                child2[i] += (1 - abs(child2[i])) * rng.random() * 0.5
            else:
                child2[i] -= (1 - abs(child2[i])) * rng.random() * 0.5

    bias_step = 0
    for i in range(379):
        if bias_step != 15:
            if child1[i] > 1:
                child1[i] = 1
            if child2[i] > 1:
                child2[i] = 1
            if child1[i] < -1:
                child1[i] = -1
            if child2[i] < -1:
                child2[i] = -1
            bias_step += 1
        else:
            if child1[i] > 10:
                child1[i] = 10
            if child2[i] > 10:
                child2[i] = 10
            if child1[i] < -10:
                child1[i] = -10
            if child2[i] < -10:
                child2[i] = -10
            bias_step = 0

    return child1, child2


def max_fitness():
    pass


if __name__ == "__main__":
    print("Ran from geneticAlgorithm.py")

    crossover_chance = 0.7
    mutation_chance = 0.3
    number_elites = 3
    tournament_size = 2
    number_generations = 1000
    board_size = 4
    games_to_run = 10

    agent_layers = [read_agent("agent" + str(i)) for i in range(51)]
    # neural_networks = [NeuralNetwork(layer[0], layer[1]) for layer in agents]
    # print([nn.to_list() for nn in neural_networks])

    run_algorithm(agent_layers)
    copy_over_agents()

    print("Managed to return!")
