
import random
import operator
from math import ceil
from statistics import pstdev

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
    def __init__(self, routes:list):
        '''The population of possible routes the salesman can take
        param cities list: list of cities for the route'''
        self.individuals = routes
        self.original_size = len(self.individuals)
        self.f_evals = 0


    def add(self, pop):
        '''adds another population to this population'''
        self.individuals.extend(pop.individuals)


    def random_individual(self):
        '''Returns a random individual from the population'''
        return random.choice(self.individuals)


    def evaluate(self) -> int:
        '''Runs the objective function on the individuals in place
        Returns the number of times the objective function was run'''
        count = 0
        for i in self.individuals:
            if i._fitness is None:
                i.calc_fitness()
                count += 1
            else:
                pass
        self.f_evals += count
        return count


    def rank(self):
        '''Ranks the routes within this population'''
        self.individuals.sort(key=operator.attrgetter('fitness'), reverse=True)


    def copy(self):
        '''Returns a copy of this population
        Each individual will be copied'''
        new_inds = []
        for ind in self.individuals:
            new_inds.append(Route(ind.genes.copy()))
        return Population(new_inds)


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
    def mean_individual(self):
        '''Returns the individual route with the mean fitness in this population'''
        ranked = self.ranked
        num = len(ranked)
        mean_idx = int(num/2)
        return ranked[mean_idx]


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


    def get_percentile(self, k):
        '''returns the distance of the kth percentile individual'''
        index = ceil(k * len(self.individuals))
        return [r.distance for i,r in enumerate(self.individuals) if i == index][0]


    def get_standard_deviation(self):
        '''Returns the standard deviation of the population's fitness'''
        fitnesses = [i.fitness for i in self.individuals]
        return pstdev(fitnesses)


    def serialize(self):
        '''Export self as json file'''
        return self.__dict__ #{'routes': len(self.routes), 'cities': len(self.routes[0])}


    def __repr__(self):
        return f'Pop; routes: {len(self.individuals)}; cities: {len(self.individuals[0])}'


def random_population(cities, size):
    '''Creates a random population of routes
    cities: cities for the routes
    size: number of routes in the population'''
    individuals = []
    for i in range(size):
        newroute = Route(cities).randomize()
        individuals.append(newroute)
    return Population(individuals)