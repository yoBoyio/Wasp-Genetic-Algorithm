import numpy as np
import math
from numpy import genfromtxt
import ga
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
# import dataset
initial_nests = genfromtxt('wasps.csv', delimiter=',')
nests = initial_nests.copy()
map_size = [100, 100]


def count_distance(position1, position2):
    return math.sqrt((position1[0]-position2[0])**2 + (position1[1]-position2[1])**2)


def count_max__nest_distance_of_current_map():
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
    nests = initial_nests.copy()
    bomb1 = X[:2]
    bomb2 = X[2:4]
    bomb3 = X[4:6]
    kill_sum = np.sum([count_bomb_kills(bomb1),
                       count_bomb_kills(bomb2), count_bomb_kills(bomb3)])
    return kill_sum


def calculate_bounds(bounds):
    min_bounds = [0]*6
    max_bounds = []
    for i in range(6):
        if i % 2 == 0:  # Χ
            max_bounds.append(map_size[0])
        else:  # Υ
            max_bounds.append(map_size[1])
    return {'min_bounds': min_bounds, 'max_bounds': max_bounds}


print('total wasps: ', sum(nests[:, 0]))

DMAX = count_distance([0, 0], map_size)

# set seed
np.random.seed(120)
start = time.time()
best_solution, best_score = ga.run(score_function=f, total_variables=6, bounds=calculate_bounds(map_size),
                                   population_size=100, generations=250, mutation_rate=0.2, mutation_step_size=0.2, gamma=0.4,
                                   beta=0.6, elitism_rate=0.1, children_population_rate=1)
end = time.time()

print(f'best score: {best_score}')
print(f'best solution: {best_solution}')
print(f'total time: {round(end-start,2)} seconds')

fig, ax = plt.subplots()
# display nesrs
for row in nests:
    x = row[1]
    y = row[2]
    ax.scatter(x, y, s=150, alpha=0.5, c='green')

# display bombs
for i in range(0, len(best_solution)-1, 2):
    x = best_solution[i]
    y = best_solution[i+1]
    ax.scatter(x, y, s=250, c="red")

ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.set_title("Bombs and nests")
ax.grid(True)
fig.tight_layout()
plt.show()
