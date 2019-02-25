
import random

from GAlgorithm import AbstractIndividual, Population, GeneticAlgorithm

class Individual(AbstractIndividual):
    '''List of genes I guess'''

    def compute_fitness(self, target):
        '''Computes fitness of the string compared to the base string'''
        fitness = 0
        for i,g in enumerate(self.genes):
            if g.value == target[i]:
                fitness += 1
        self.fitness = fitness
        return fitness


    def __repr__(self):
        return ''.join([g.value for g in self.genes])

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


def select(population, tourny_size:int=2):
    '''Selects a parent population from the population
    Uses a tournament to select parents'''
    parents = []
    for _ in population.individuals:
        winner = population.random_individual()
        for _ in range(1, tourny_size):
            competitor = population.random_individual()
            if competitor.fitness > winner.fitness:
                winner = competitor
        parents.append(winner)
    return Population(parents)


def crossover(parents):
    '''Crosses the parents over to create children'''
    children = []
    for i, individual in enumerate(parents.individuals):
        children.append(cross(individual, parents.individuals[-i-1]))
    return Population(children)


def cross(p1, p2):
    # number of genes from 1 parent
    num_from_p1 = random.randint(0, len(p1.genes))
    idx_from_p1 = random.sample(range(len(p1.genes)), k=num_from_p1)
    genes_from_p1 = [p1.genes[x] for x in idx_from_p1]

    # get the rest from the other parent
    idx_from_p2 = [i for i in range(len(p1.genes)) if (i not in idx_from_p1)]
    genes_from_p2 = [p2.genes[x] for x in idx_from_p2]

    # combine p1 and p2 genes
    idx = idx_from_p1 + idx_from_p2
    scrambled_genes = genes_from_p1 + genes_from_p2

    # sorted genes out
    child_genes = [g for _,g in sorted(zip(idx, scrambled_genes))]
    return Individual(child_genes)


def mutate(children, chance):
    '''Mutates the children, given the chance
    children: list of route children
    chance: chance of mutation btw 0 and 1'''
    for child in children.individuals:
        mutate_1(child, chance)
    return Population(children.individuals)


def mutate_1(child, chance):
    '''Mutates each gene in child with a chance of chance'''
    if random.random() < chance:
        gene = random.choice(child.genes)
        gene = gene.copy()
        gene.mutate()
        child.clear_fitness()

    
def initialize_pop(size, target, allowed_params):
    '''Initialize the population'''
    individuals = []
    for _ in range(size):
        genes = [Gene(allowed_params) for _ in target]
        [gene.mutate() for gene in genes]
        individuals.append(Individual(genes))
    pop = Population(individuals)
    return pop


def run():
    '''runs a new string pattern matching GA'''
    vals = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !,.'
    target = 'Hello world, I currently work.'
    init_pop = initialize_pop(500, target, vals)
    ga = GeneticAlgorithm(init_pop, 2, 0.02, 30000)
    ga.fitness_params = {target}
    ga.select = select
    ga.crossover = crossover
    ga.mutate = mutate
    ga.run_without_ml()

    for p in ga.pop_history:
        print('{}, {}'.format(p.best_individual.fitness, p.best_individual))

if __name__ == "__main__":
    run()