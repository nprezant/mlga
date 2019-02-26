# -*- coding: utf-8 -*-

import random

from GAlgorithm import AbstractIndividual, Population

from .population import Route


def crossover(parents):
    '''Crosses the parents over to create children'''
    children:Route = []
    for i, individual in enumerate(parents.individuals):
        children.append(cross(individual, parents.individuals[-i-1]))
    return Population(children)


def cross(p1, p2):
    '''Crosses parent1 with parent2
    Note that all cities must be represented'''
    
    child = []

    r1 = random.randint(0, len(p1.genes))
    r2 = random.randint(0, len(p1.genes))

    start = min(r1, r2)
    stop = max(r1, r2)

    cross_segment = p1.genes[start:stop]
    needed_cities = [city for city in p1.genes if city not in cross_segment]

    child.extend(cross_segment)

    for _ in range(len(needed_cities)):
        r = random.randint(0, len(needed_cities)-1)
        child.append(needed_cities.pop(r))

    return Route(child)


def mutate(children, chance):
    '''Mutates the children, given the chance
    children: list of route children
    chance: chance of mutation btw 0 and 1'''

    for child in children.individuals:
        mutate_2(child, chance)
    return Population(children.individuals)


def mutate_1(child, chance):
    '''Mutates each gene in child with a chance of chance'''
    for i in range(len(child.genes)):
        if random.random() < chance:
            r = random.randint(0, len(child.genes)-1)
            child.genes[i], child.genes[r] = child.genes[r], child.genes[i]
            child.clear_fitness()


def mutate_2(child, chance):
    '''Mutates each child with a chance of self.chance'''
    if random.random() < chance:
        A = random.randint(0, len(child.genes)-1)
        B = random.randint(0, len(child.genes)-1)
        child.genes[A], child.genes[B] = child.genes[B], child.genes[A]
        child.clear_fitness()


def stochastic_acceptance_selection(population):
    '''Selects parents based on stochastic acceptance.
    1) Randomly select individual
    2) Accept selection with probability fi/fm
    where fm = maximum population fitness'''
    max_fitness = population.max_fitness
    min_fitness = population.min_fitness
    parents = []
    complete = False
    while not complete:
        individual = population.random_individual()
        probality = (individual.fitness - min_fitness) / max_fitness
        if random.random() <= probality:
            parents.append(individual)
            if len(parents) == len(population.individuals):
                complete = True
    return Population(parents)