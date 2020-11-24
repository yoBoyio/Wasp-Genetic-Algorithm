import numpy as np
import operator
"""
run(score_function, total_variables, bounds, population_size,
        generations, children_population_rate=1,
        gamma=0.1, mutation_rate=0.1, mutation_step_size=0.1, beta=1)

use
---
Run the genetic algorithm

Parameters
----------

score_function: function
    the evaluation function

total_variables: int
    variables (solution) to generate

bounds: dict{'min_bounds': Number,'max_bounds': Number}
    bounds for variables

population_size: int

generations: int
    Total iterations

children_population_rate: [0,1]
    How big should the child population be compared to the original

gamma: [0,1]
    used in crossover to generate alpha.
    It manipulates the variance of alpha.

mutation_rate: [0,1]
    how often should a child be mutated (that sounds wrong)

mutation_step_size: 
    step size for mutation modification:
    bigger step-size finds a solution quicker, however it may not be the best one

beta: [0,1]
    modifier for roulette.
    Higher beta gives more probability to higher solutions

elitism_rate: [0,1]
    Percentage of current population members to be passed onto the next one

Returns
-------
best_solution: [Number]
best_score: int

"""


def run(score_function, total_variables, bounds, population_size,
        generations, children_population_rate=1, gamma=0.1,
        mutation_rate=0.1, mutation_step_size=0.1, beta=1, elitism_rate=0.1):

    # initialize best solution

    best_solution = [0]*total_variables
    best_score = 0
# initialize  population
    population = []
    for _ in range(population_size):
        # random position inside bounds
        position = np.array(np.random.uniform(
            bounds['min_bounds'], bounds['max_bounds'], total_variables))
        # calculate score
        score = score_function(position)
        population.append({
            'position': position,
            'score': score
        })

    # update best score, best solution
    for solution in population:
        if solution['score'] > best_score:
            best_score = solution['score']
            best_solution = solution['position']

# Iterate Generations
# calculate children population using given rate
    total_children = int(
        np.round(children_population_rate*population_size))
    for g in range(generations):

        # calculate average previous generation score
        scores = np.array([x['score'] for x in population])
        avg_score = np.mean(scores)
        if avg_score != 0:  # normalization
            scores /= avg_score
        probabilities = np.exp(-beta*scores)

        print(
            f'generation: {g} / {generations}, best score: {best_score}')

        children_population = []
        for _ in range(total_children//2):  # /2 since we generate 2 children per iteration
            parent_1 = population[roulette_selection(probabilities)]
            parent_2 = population[roulette_selection(probabilities)]

            # Generate offsprings
            child_1, child_2 = generate_children(
                parent_1, parent_2, gamma, mutation_rate, mutation_step_size, bounds)

            # Evaluate offsprings
            best_score, best_solution = evaluate_child(
                child_1, child_2, best_score, best_solution, score_function)

            # Add offsprings to child population
            children_population += [child_1, child_2]
    # merge populations via elitism
        # sort population
        population = sorted(population, key=lambda y: y['score'], reverse=True)

        # pass the elites to the new population
        total_elites = int(elitism_rate*population_size)
        new_population = population[0:total_elites]

        children_population = sorted(
            children_population, key=lambda y: y['score'], reverse=True)

    # pass the best children to the next
        # if population_size = children_population_size,remove elite length
        if(children_population_rate == 1):
            new_population += children_population[0:int(
                (1-elitism_rate)*population_size)]
        # population_size != children_population_size, therefore add every child + rest of population best
        else:
            new_population += children_population
            population_rest_best_size = population_size-total_children
            new_population += population[total_elites: population_rest_best_size]
        
        population = new_population
    return best_solution, best_score


def apply_bounds(x, min_bounds, max_bounds):
    # if position is out of map, return the map bounds
    x['position'] = np.minimum(x['position'], max_bounds)
    x['position'] = np.maximum(x['position'], min_bounds)
    return x


def generate_children(parent_1, parent_2, gamma, mutation_rate, mutation_step_size, bounds):
    # Crossover
    child_1, child_2 = crossover(parent_1, parent_2, gamma)

    # mutation
    child_1 = mutate(child_1, mutation_rate, mutation_step_size)
    child_2 = mutate(child_2, mutation_rate, mutation_step_size)

    # Refactor position via bounds
    child_1 = apply_bounds(
        child_1, bounds['min_bounds'], bounds['max_bounds'])
    child_2 = apply_bounds(
        child_2, bounds['min_bounds'], bounds['max_bounds'])
    return child_1, child_2


def evaluate_child(child_1, child_2, best_score, best_solution, score_function):
    # Evaluate first offspring
    child_1['score'] = score_function(child_1['position'])
    if child_1['score'] > best_score:
        best_score = child_1['score']
        best_solution = child_1['position']

    # Evaluate second offspring
    child_2['score'] = score_function(child_2['position'])
    if child_2['score'] > best_score:
        best_score = child_2['score']
        best_solution = child_2['position']

    return best_score, best_solution


def mutate(x, mutation_rate, mutation_step_size):
    y = x.copy()
    # choose a random number [0,1] for each element of x position and compare it to the mutation rate
    # each lower element than mutation rate is flagged with True
    flag = np.random.rand(*x['position'].shape) <= mutation_rate
    # find the indexes of  True flagged elements
    indexes = np.argwhere(flag)
    # mutate True flagged elements
    y['position'][indexes] += mutation_step_size * \
        np.random.randn(*indexes.shape)
    return y


# single point crossover
def crossover(p1, p2, gamma):
    child_1 = p1.copy()
    child_2 = p1.copy()
    # len(alpha) = total_variablesj
    alpha = np.random.uniform(-gamma, 1+gamma, *child_1['position'].shape)
    child_1['position'] = alpha*p1['position'] + (1-alpha)*p2['position']
    child_2['position'] = alpha*p2['position'] + (1-alpha)*p1['position']
    return child_1, child_2


def roulette_selection(probabilities):
    c = np.cumsum(probabilities)
    r = sum(probabilities)*np.random.rand()
    indexes = np.argwhere(r <= c)
    # Return the first chosen index
    return indexes[0][0]


