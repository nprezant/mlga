# -*- coding: utf-8 -*-

import random

from GAlgorithm import AbstractIndividual, Population


class Route(AbstractIndividual):
    '''Route bro'''

    def randomize(self):
        '''Randomizes the route'''
        self._genes = random.sample(self._genes, len(self._genes))
        return self


    @property
    def x(self):
        '''returns x as a list of city data x coordinates'''
        return [city.x for city in self.genes] + [self.genes[0].x]


    @property
    def y(self):
        '''returns y as a list of city data y coordinates'''
        return [city.y for city in self.genes] + [self.genes[0].y]


    def make_training_data(self):
        '''Converts itself into a list of number to be used as training data'''
        return self.x + self.y


def distance(city_list):
    '''The distance of this route'''
    d = 0
    for i in range(len(city_list)-1):
        d += city_list[i].distance_to(city_list[i+1])
    d += city_list[-1].distance_to(city_list[0])
    return d


def compute_fitness(route):
    '''Calculates fitness of this route on scale of 0 to 1
    Need a function to do so to keep track of the
    number of times this function is called'''
    route.distance = distance(route.genes)
    fitness = 1 / route.distance
    return fitness
    


class City:
    def __init__(self, x=0, y=0):
        '''A city that the salesman must travel to'''
        self.x = x
        self.y = y


    def distance_to(self, other):
        '''Distance to another city'''
        return ((self.x-other.x)**2 + (self.y-other.y)**2)**(1/2)


    def randomize(self, xmin, xmax, ymin, ymax):
        '''Randomizes the city's location'''
        self.x = random.randint(xmin, xmax)
        self.y = random.randint(ymin, ymax)
        return self


    def serialize(self):
        '''Export self as json file'''
        return {'x': self.x, 'y': self.y}
    

    def __str__(self):
        return f'City ({self.x}, {self.y})'


def random_population(cities, size):
    '''Creates a random population of routes
    cities: cities for the routes
    size: number of routes in the population'''
    individuals = []
    for i in range(size):
        newroute = Route(cities).randomize()
        individuals.append(newroute)
    return Population(individuals)


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