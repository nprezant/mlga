
import random
import operator

OBJECTIVE_CALLS = 0

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


class Route:
    def __init__(self, ordered_cities):
        '''The route the salesman travels through the cities'''
        self._genes = ordered_cities
        self._fitness = None


    def randomize(self):
        '''Randomizes the route'''
        self._genes = random.sample(self._genes, len(self._genes))
        return self


    @property
    def genes(self):
        return self._genes


    @genes.setter
    def genes(self, val):
        self._genes = val
        self._fitness = None


    @property
    def distance(self):
        '''The distance of this route'''
        d = 0
        for i in range(len(self.genes)-1):
            d += self.genes[i].distance_to(self.genes[i+1])
        d += self.genes[-1].distance_to(self.genes[0])
        return d


    @property
    def fitness(self):
        '''ranks the fitness of this route on scale of 0 to 1'''
        if self._fitness is None:
            self._fitness = 1 / self.distance
            OBJECTIVE_CALLS += 1
        return self._fitness


    @property
    def x(self):
        '''returns x as a list of city data to plot'''
        return [city.x for city in self.genes] + [self.genes[0].x]


    @property
    def y(self):
        '''returns y as a list of city data to plot'''
        return [city.y for city in self.genes] + [self.genes[0].y]


    def serialize(self):
        '''Export self as json dump compatible'''
        d = self.__dict__
        d['__Route__'] = True
        return d


    def __len__(self):
        return len(self.genes)


    def __str__(self):
        return f'(d={self.distance}) (f={self.fitness})'


    def __repr__(self):
        return [c for c in self.genes]


class Population:
    def __init__(self, routes:list, size=None):
        '''The population of possible routes the salesman can take
        param cities list: list of cities for the route'''
        self.individuals = routes
        if size is None:
            self.size = len(self.individuals)
        else:
            self.size = size


    def randomize(self, cities, size):
        '''Randomizes (resets) the population of routes
        param size int: size of population'''
        self.individuals = []
        self.size = size
        for i in range(size):
            newroute = Route(cities).randomize()
            self.individuals.append(newroute)
        return self


    def add(self, pop):
        '''adds another population to this population'''
        self.individuals.extend(pop.individuals)


    def random_individual(self):
        '''Returns a random individual from the population'''
        return random.choice(self.individuals)


    def rank(self):
        '''Ranks the routes within this population'''
        self.individuals.sort(key=operator.attrgetter('fitness'), reverse=True)


    @property
    def ranked(self):
        '''Returns the ranked routes, but doesn't change the internal state'''
        return sorted(self.individuals, key=operator.attrgetter('fitness'), reverse=True)


    @property
    def genes(self):
        '''Returns a copied list of the cities in the first route'''
        return self.individuals[0].genes.copy()


    @property
    def best_individual(self):
        '''Returns the individual route with the best fitness in this population'''
        return max(self.individuals, key=operator.attrgetter('fitness'))


    @property
    def max_fitness(self):
        '''Finds the maximum fitness route of the population'''
        return max(self.individuals, key=operator.attrgetter('fitness')).fitness


    @property
    def min_fitness(self):
        '''Finds the minimum fitness route of the population'''
        return min(self.individuals, key=operator.attrgetter('fitness')).fitness


    @property
    def mean_fitness(self):
        '''Finds the mean fitness of the population'''
        fitnesses = [i.fitness for i in self.individuals]
        return sum(fitnesses) / len(fitnesses)


    def serialize(self):
        '''Export self as json file'''
        return self.__dict__ #{'routes': len(self.routes), 'cities': len(self.routes[0])}


    def __repr__(self):
        return f'Pop; routes: {len(self.individuals)}; cities: {len(self.individuals[0])}'