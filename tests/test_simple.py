
import random

from GAlgorithm import AbstractIndividual, Population, GeneticAlgorithm

class Sentence(AbstractIndividual):
    '''List of letters that form a sentence'''

    def compute_fitness(self, target):
        '''Computes fitness of the string compared to the base string'''
        fitness = 0
        for i,g in enumerate(self.genes):
            if g.value == target[i]:
                fitness += 1
        self.fitness = fitness
        return fitness


    def __repr__(self):
        return ''.join([str(g) for g in self.genes])

    def __str__(self):
        return self.__repr__()


class Gene:
    def __init__(self, rng:list, val=None):
        self.value = val
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

    
def initialize_pop(size, target, allowed_params):
    '''Initialize the population'''
    individuals = []
    for _ in range(size):
        genes = [Gene(allowed_params) for _ in target]
        [gene.mutate() for gene in genes]
        individuals.append(Sentence(genes))
    pop = Population(individuals)
    return pop


def run():
    '''runs a new string pattern matching GA'''
    vals = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !,.'
    target = 'Hello world, I mostly work.'
    init_pop = initialize_pop(500, target, vals)
    ga = GeneticAlgorithm(init_pop, 2, 0.05, 30000)
    ga.fitness_params = {target}
    ga.run_without_ml()

    for p in ga.pop_history:
        print('{}, {}'.format(p.best_individual.fitness, p.best_individual))

if __name__ == "__main__":
    run()