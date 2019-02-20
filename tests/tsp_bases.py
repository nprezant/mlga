# -*- coding: utf-8 -*-

import random

from GAlgorithm import AbstractIndividual, Population


class Route(AbstractIndividual):
    '''Route bro'''

    def randomize(self):
        '''Randomizes the route'''
        self._genes = random.sample(self._genes, len(self._genes))
        return self


    def copy(self):
        '''Copies this individual'''
        return Route(self.genes.copy())


    @property
    def distance(self):
        '''The distance of this route'''
        d = 0
        for i in range(len(self.genes)-1):
            d += self.genes[i].distance_to(self.genes[i+1])
        d += self.genes[-1].distance_to(self.genes[0])
        return d


    def calc_fitness(self):
        '''Calculates fitness of this route on scale of 0 to 1
        Need a function to do so to keep track of the
        number of times this function is called'''
        self._fitness = 1 / self.distance


    @property
    def fitness(self):
        '''retreives the fitness of this route.
        You must call the "calc_fitness" method before
        accessing this property'''
        assert self._fitness is not None
        return self._fitness


    @property
    def x(self):
        '''returns x as a list of city data to plot'''
        return [city.x for city in self.genes] + [self.genes[0].x]


    @property
    def y(self):
        '''returns y as a list of city data to plot'''
        return [city.y for city in self.genes] + [self.genes[0].y]
    


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


    def __repr__(self):
        return f'City({self.x},{self.y})'


def random_population(cities, size):
    '''Creates a random population of routes
    cities: cities for the routes
    size: number of routes in the population'''
    individuals = []
    for i in range(size):
        newroute = Route(cities).randomize()
        individuals.append(newroute)
    return Population(individuals)