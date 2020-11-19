import numpy as np
import operator


def run(score_function, total_variables, bounds, population_size,
        generations, children_population_rate=1,
        gamma=0.1, mutation_rate=0.1, sigma=0.1, beta=1):

    best_solution = [0]*total_variables
    best_score = 0

# initialize population
    min_bounds = [0]*total_variables
    max_bounds = []
    for i in range(total_variables):
        if i % 2 == 0:  # Χ
            max_bounds.append(bounds[0])
        else:  # Υ
            max_bounds.append(bounds[1])

    # min bound,max bound, population shape
    population = []
    for _ in range(population_size):
        position = np.array(np.random.uniform(
            min_bounds, max_bounds, total_variables))
        score = score_function(position)
        population.append({
            'position': position,
            'score': score
        })

    # initialize best score, best solution
    for solution in population:
        if solution['score'] > best_score:
            best_score = solution['score']
            best_solution = solution['position']

# Iterate Generations
    total_children = int(
        np.round(children_population_rate*population_size/2)*2)
   # print('pop len:', len(population['scores']))
    for g in range(generations):

        scores = np.array([x['score'] for x in population])
        avg_score = np.mean(scores)
        if avg_score != 0:
            scores = scores / avg_score
        probabilities = np.exp(-beta*scores)

        print(
            f'generation: {g} / {generations}, average score: {np.round(avg_score)},best score: {best_score}')

        children_population = []
        for i in range(total_children//2):
            parent_1 = population[roulette_selection(probabilities)]
            parent_2 = population[roulette_selection(probabilities)]

            # Crossover
            child_1, child_2 = crossover(parent_1, parent_2, gamma)

            # mutation
            child_1 = mutate(child_1, mutation_rate, sigma)
            child_2 = mutate(child_2, mutation_rate, sigma)

            # Refactor bounds
            child_1 = apply_bounds(child_1, min_bounds, max_bounds)
            child_2 = apply_bounds(child_2, min_bounds, max_bounds)

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

            children_population.append(child_1)
            children_population.append(child_2)

        population += children_population
        population = sorted(population, key=lambda y: y['score'], reverse=True)
        population = population[0:population_size]

    print(f'best_score: {best_score}')
    print(f'best_solution: {best_solution}')


def apply_bounds(x, min_bounds, max_bounds):
    x['position'] = np.minimum(x['position'], max_bounds)
    x['position'] = np.maximum(x['position'], min_bounds)
    return x


def mutate(x, mutation_rate, sigma):
    y = x.copy()
    flag = np.random.rand(*x['position'].shape) <= mutation_rate
    indexes = np.argwhere(flag)
    y['position'][indexes] += sigma*np.random.randn(*indexes.shape)
    return y


def crossover(p1, p2, gamma):
    child_1 = p1.copy()
    child_2 = p1.copy()
    alpha = np.random.uniform(-gamma, 1+gamma, *child_1['position'].shape)
    child_1['position'] = alpha*p1['position'] + (1-alpha)*p2['position']
    child_2['position'] = alpha*p2['position'] + (1-alpha)*p1['position']
    return child_1, child_2


def roulette_selection(probabilities):
    c = np.cumsum(probabilities)
    r = sum(probabilities)*np.random.rand()
    indexes = np.argwhere(r <= c)
    return indexes[0][0]
