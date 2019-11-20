
import random

from GAlgorithm import (
    Population, 
    GeneticAlgorithm, 
    initialize_population,
    fitness_plot)


def sum_knapsack(knapsack, items):
    '''Finds the total value and weight of a knapsack'''
    total_value = 0
    total_weight = 0
    for i,g in enumerate(knapsack.genes):
        
        if g.value == True:
            # only count genes that are included
            total_value += items[i].value
            total_weight += items[i].weight
    
    return total_value, total_weight


def fitness(knapsack, items, max_weight):
    '''Computes fitness of the string compared to the base string'''
    
    value, weight = sum_knapsack(knapsack, items)
    
    if weight > max_weight:
        # can't carry something this heavy
        value = 0

    return value


def training_data(knapsack):
    '''Turns a knapsack into classifier-digestable training data'''
    return [int(use_item.value) for use_item in knapsack.genes]


class Item:

    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
    
    def __str__(self):
        return f'value: {self.value}, weight: {self.weight}'


def run():
    '''runs a new knapsack problem GA'''
    items = [Item(random.randint(1,30), random.randint(1,30)) for _ in range(30)]
    max_weight = 30

    print([str(i) for i in items])

    vals = [True, False]
    init_pop = initialize_population(500, 30, vals, default_val=False)
    for sack in init_pop.individuals:
        some_items = random.sample(sack.genes, 2)
        for item in some_items: item.value = True

    ga = GeneticAlgorithm(init_pop, fitness, 2, 0.05, 20000)
    ga.fitness_params = {'items':items, 'max_weight':max_weight}

    ga.run_without_ml()
    hist1 = (ga.pop_history.copy(), 'Without ML')
    for p in ga.pop_history:
        value, weight = sum_knapsack(p.best_individual, items)
        print('Knapsack: value={}, weight={}'.format(value, weight))

    print('now WITH MACHINE LEARNING')
    ga.training_data_function = training_data
    ga.classifier_percentage = 0.2
    ga.run_with_ml()
    hist2 = (ga.pop_history.copy(), 'With ML')
    for p in ga.pop_history:
        value, weight = sum_knapsack(p.best_individual, items)
        print('Knapsack: value={}, weight={}'.format(value, weight))

    fitness_plot([hist1, hist2], 'Knapsack Fitness Plot')


if __name__ == "__main__":
    run()