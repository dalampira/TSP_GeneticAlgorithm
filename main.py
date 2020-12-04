import random
import copy
import math
import matplotlib.pyplot as plt


matrix = []
population = []
cities = []
values = []
best = []
best_value = None
current_best = []
evolving_best_values = []

CROSSOVER_PROBABILITY = 0.7
GENERATIONS = 3000


#Here, we create a matrix where we store the distance between all the cities
#We undestand that the matrix is symmetrical because we have specific coordinates that don't change
def create_distance_matrix(dataset):
    global matrix
    length = len(dataset)
    matrix = [[0 for x in range(length)] for y in range(length)]
    for i in range(0,length):
        x_i = float(dataset[i][1])
        y_i = float(dataset[i][2])
        for j in range(0,length):
            x_j = float(dataset[j][1])
            y_j = float(dataset[j][2])
            matrix[i][j] = math.sqrt((abs(x_i - x_j)**2 + abs(y_i - y_j)**2))
            matrix[j][i] = math.sqrt((abs(x_i - x_j)**2 + abs(y_i - y_j)**2))


#We evaluate the total distance of all the routes of the population by adding the
#distance between every city of the route. The smallest distance is the best one.
def evaluate(route):
    route_length = len(route)
    total_distance = 0
    for i in range(0, route_length):
        if i+1 == route_length:
            total_distance += matrix[route[i]][route[0]]
        else:
            total_distance += matrix[route[i]][route[i + 1]]
    return total_distance


#We store the best distance value of every generation, so as to plot our program's
#evolution in the end. And we return this value and the index of the best route
def get_current_best():
    global evolving_best_values
    best_parent_index = None
    current_best_value = min(values)
    evolving_best_values.append(current_best_value)

    for i in range(0,len(population)):
        if values[i]==current_best_value:
            best_parent_index = i
    return  best_parent_index, current_best_value


#After evaluating every route of the population, we store the best route
#and the distance value of this route
def set_best_value():
    global current_best, best_value, best, values

    for i in range(len(population)):
        values[i] = evaluate(population[i])

    current_best = get_current_best()
    if (best_value is None) or (best_value > current_best[1]):
        best = population[current_best[0]]
        best_value = current_best[1]


#We generate random numbers so as to mutate two cities of the route with each other
def mutate(route):
    index1 = random.randint(0, len(route)-1)
    index2 = random.randint(0, len(route)-1)

    while index1 == index2:
        index2 = random.randint(0, len(route)-1)
    city1 = route[index1]
    city2 = route[index2]

    route[index1] = city2
    route[index2] = city1

    return route


#We generate two random points so as to get the children of the two parents.
#We append the cities between these numbers to the first child and the others
#to the other child. From the other parent, we append the cities that are not
#already included to each child. Finally, we return the children.
def get_children(x, y):

    children = []
    child1 = []
    child2 = []
    point1 = random.randint(0, len(population[x]))
    point2 = random.randint(0, len(population[y]))

    startPoint = min(point1, point2)
    endPoint = max(point1, point2)

    for i in range(startPoint, endPoint):
        child1.append(population[x][i])

    for item in population[y]:
        if item not in child1:
            child1.append(item)
    children.append(child1)

    for i in range(0, startPoint):
        child2.append(population[x][i])
    for i in range(endPoint, len(population[x])):
        child2.append(population[x][i])
    for item in population[y]:
        if item not in child2:
            child2.append(item)
    children.append(child2)
    return children


#We get the children and we mutate both of them. After that, we evaluate the routes of both the children
#and the parents to check which two are the best ones that they will continue to the next generation.
#If a child is among the best ones, we check which parent was left out, so as to store it in his position.
#If a parent is among the best ones, we don't need to do anything because he is already stored in the population.
def crossover(x, y):
    global population
    children = get_children(x,y)

    children[0] = mutate(children[0])
    children[1] = mutate(children[1])
    next_gen = list()

    ev_child1 = evaluate(children[0])
    next_gen.append(ev_child1)
    ev_child2 = evaluate(children[1])
    next_gen.append(ev_child2)

    ev_parent1 = evaluate(population[x])
    next_gen.append(ev_parent1)
    ev_parent2 = evaluate(population[y])
    next_gen.append(ev_parent2)

    replaced = list()
    replaced.append(max(next_gen))
    next_gen.remove(max(next_gen))
    replaced.append(max(next_gen))
    next_gen.remove(max(next_gen))

    for i in range(0,len(next_gen)):
        if next_gen[i] == ev_child1:
            for j in range(0,len(replaced)):
                if replaced[j] == ev_parent1:
                    population[x] = children[0]
                if replaced[j] == ev_parent2:
                    population[y] == children[0]

        if next_gen[i] == ev_child2:
            for j in range(0, len(replaced)):
                if replaced[j] == ev_parent1:
                    population[x] = children[1]
                if replaced[j] == ev_parent2:
                    population[y] == children[1]


#Based on the crossover propability, we generate which parents are going
#to do a crossover and we perform a crossover with each other.
def crossover_population():
    queue = []
    for pop in range(len(population)):
        if random.random() < CROSSOVER_PROBABILITY:
            queue.append(pop)
    random.shuffle(queue)
    for i in range(0, len(queue) - 1, 2):
        crossover(queue[i], queue[i + 1])


#We keep the best route of the current generation and the best route in general to be
#parents for the next generation and we randomly append the other parents too.
def select_parents():
    global population
    parents = []
    parents.append(population[current_best[0]])
    parents.append(copy.deepcopy(best))

    for i in range(2, len(population)):
        randomNum = random.randint(0,(len(population)-1))
        parents.append(population[randomNum])
    population = parents


#For every new generation, we select the parents, do crossover and store the
#best distance and route
def new_generation():
    select_parents()
    crossover_population()
    set_best_value()


#We create the population by randomly generating 100 routes including all
#the cities we were given. Each of them can appear only once
def create_population():
    for i in range(0, 100):
        route = random.sample(cities, len(cities))
        population.append(route)


#We initialize everything by creating the distance matrix, a list of the cities, the initial population
#and getting the best distance of the routes
def initialize():
    global values
    create_distance_matrix(dataset)

    for line in dataset:
        cities.append(int(line[0])-1)

    create_population()
    values = [0 for i in range(0,len(population))]
    set_best_value()


#We calculated the best route and we want this route to start from the first city,
#so we need to make sure that we start from that city
def final_route(route):
    final_route = []
    for i in range(0, len(route)):
        if route[i] == 0:
            index_i = i
    for i in range(index_i, len(route)):
        final_route.append(route[i])
    for i in range(0, index_i):
        final_route.append(route[i])
    return final_route


#We open the file and to store the data
def open_file():
    with open('berlin52.tsp') as fp:
        dataset = [line.split() for line in fp.readlines() if line]

    return dataset




dataset = open_file()
initialize()
print("Initial distance: " + str(best_value))
for i in range(0, GENERATIONS):
    new_generation()
print("Final distance: " + str(best_value))

print(final_route(best))

plt.plot(evolving_best_values)
plt.ylabel('Distance')
plt.xlabel('Generations')
plt.show()
