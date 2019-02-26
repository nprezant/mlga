
import random

from GAlgorithm import AbstractIndividual, Population, GeneticAlgorithm

class Knapsack(AbstractIndividual):
    '''List of letters that form a sentence'''

    def compute_fitness(self, items, max_weight):
        '''Computes fitness of the string compared to the base string'''
        total_value = 0
        total_weight = 0
        for i,g in enumerate(self.genes):
            
            if g.value == True:
                # only count genes that are included
                total_value += items[i].value
                total_weight += items[i].weight
        
        if total_weight > max_weight:
            # can't carry something this heavy
            total_value = 0

        self.fitness = total_value
        return total_value


class Gene:
    def __init__(self, rng:list, value=None):
        self.value = value
        self.rng = rng


    def mutate(self):
        ''' pick random value from the list of allowed values'''
        self.value = random.choice(self.rng)


    def copy(self):
        '''Makes a copy of itself'''
        return Gene(self.rng, self.value)


    def __str__(self):
        return str(self.value)


    def __repr__(self):
        return self.__str__()


class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

    
    def __str__(self):
        return f'value: {self.value}, weight: {self.weight}'

    
def initialize_pop(pop_size, indiv_size, allowed_params):
    '''Initialize the population'''
    individuals = []
    for _ in range(pop_size):
        genes = [Gene(allowed_params) for _ in range(indiv_size)]
        for gene in genes:
            gene.value = False
        individuals.append(Knapsack(genes))
    pop = Population(individuals)
    return pop


def run():
    '''runs a new string pattern matching GA'''
    items = [Item(random.randint(1,30), random.randint(1,30)) for _ in range(30)]
    max_weight = 30

    print([str(i) for i in items])

    vals = [True, False]
    init_pop = initialize_pop(500, 30, vals)
    ga = GeneticAlgorithm(init_pop, 2, 0.05, 20000)
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

        print('{}, {}, {}'.format(total_value, total_weight, p.best_individual))

if __name__ == "__main__":
    run()