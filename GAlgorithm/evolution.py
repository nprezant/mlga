
import random

from .population import Population


def tournament_selection(population, tourny_size:int=2):
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


def order_independent_crossover(parents):
    '''Crosses the parents over to create children
    Does not keep the order of the genes'''
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

    # sort genes out
    child_genes = [g for _,g in sorted(zip(idx, scrambled_genes))]

    # make new individual
    individ = p1.copy()
    individ.genes = child_genes
    return individ


def mutation(children, chance):
    '''Mutates the children, given the chance
    children: list of route children
    chance: chance of mutation btw 0 and 1'''
    for child in children.individuals:
        mutate_child(child, chance)
    return Population(children.individuals)


def mutate_child(child, chance):
    '''Mutates each gene in child with a chance of chance
    Requires that the gene has a mutation and copy method.'''
    if random.random() < chance:
        gene = random.choice(child.genes)
        gene = gene.copy()
        gene.mutate()
        child.clear_fitness()