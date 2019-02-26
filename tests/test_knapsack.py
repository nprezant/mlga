
import random

from GAlgorithm import Population, GeneticAlgorithm, initialize_population

def fitness(knapsack, items, max_weight):
    '''Computes fitness of the string compared to the base string'''
    total_value = 0
    total_weight = 0
    for i,g in enumerate(knapsack.genes):
        
        if g.value == True:
            # only count genes that are included
            total_value += items[i].value
            total_weight += items[i].weight
    
    if total_weight > max_weight:
        # can't carry something this heavy
        total_value = 0

    return total_value


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
    ga = GeneticAlgorithm(init_pop, fitness, 2, 0.05, 20000)
    ga.fitness_params = {'items':items, 'max_weight':max_weight}
    ga.run_without_ml()

    for p in ga.pop_history:
        total_value = 0
        total_weight = 0
        for i,g in enumerate(p.best_individual.genes):
            if g.value == True:
                # only count genes that are included
                total_value += items[i].value
                total_weight += items[i].weight

        print('Knapsack: value={}, weight={}'.format(total_value, total_weight))

if __name__ == "__main__":
    run()