

import random

from mlga import Individual, Population, Objective


Population.objective_type = Objective.MINIMIZE


class Route(Individual):
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

    def to_csv(self, fp):
        '''Writes out identifying data for this route to a CSV file'''
        with open(fp, 'w') as f:
            f.write('x\ty\n')
            for city in self.genes:
                f.write(f'{city.x}\t{city.y}\n')


def make_training_data(route):
    '''Converts the route into a list of number to be used as training data
    Also sets the x and y data for the route'''
    return route.x + route.y


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
    # fitness = 1 / route.distance
    fitness = route.distance
    return fitness
