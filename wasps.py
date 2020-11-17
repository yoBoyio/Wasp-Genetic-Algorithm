import numpy as np
import math
from geneticalgorithm import geneticalgorithm as ga
from numpy import genfromtxt

# import dataset
nests = genfromtxt('wasps.csv', delimiter=',')


def count_distance(position1, position2):
    return math.sqrt((position1[0]-position2[0])**2 + (position1[1]-position2[1])**2)


def count_max_distance():
    MAX_DISTANCE = 0
    for nest in nests:
        nest_distance = 0
        # compare current nest with every other nest
        for restnest in nests:
            # last 2 columns are X and Y coordinates
            distance = count_distance(nest[-2:], restnest[-2:])
            if distance > nest_distance:
                nest_distance = distance
        if nest_distance > MAX_DISTANCE:
            MAX_DISTANCE = nest_distance
    return MAX_DISTANCE


def count_nest_kills(bomb_location, nest):
    nest_wasps = nest[0]
    nest_location = nest[-2:]
    distance = count_distance(bomb_location, nest_location)
    score = round(nest_wasps*(DMAX/(20*distance + 0.00001)))
    # if is strong enough to kill more than given nest's wasps, then the kill count in equal to the nest wasps.
    if score > nest_wasps:
        return nest_wasps
    else:
        return score


def count_bomb_kills(bomb):
    total_kills = 0
    global nests
    for index, nest in enumerate(nests):
        kills = count_nest_kills(bomb, nest)
        total_kills += kills
        nests[index][0] -= kills
    return total_kills


def f(X):
    global nests
    nests = genfromtxt('wasps.csv', delimiter=',')
    bomb1 = X[:2]
    bomb2 = X[2:4]
    bomb3 = X[4:6]
    kill_sum = -np.sum([count_bomb_kills(bomb1),
                        count_bomb_kills(bomb2), count_bomb_kills(bomb3)])
    return kill_sum


print('total wasps: ', sum(nests[:, 0]))

DMAX = count_max_distance()

# The problem dimensions are 6, since we have 3 bombs and each bomb has 2 coordinates.
varbound = np.array([[0, 100]]*6)

algorithm_param = {'max_num_iteration': 3000,
                   'population_size': 300,
                   'mutation_probability': 0.05,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'one_point',
                   'max_iteration_without_improv': 10}

model = ga(function=f, dimension=6, variable_type='real',
           variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model.run()
