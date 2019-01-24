
import json
import random
import operator
from functools import singledispatch

from plotter import plotroutes

@singledispatch
def to_serializable(o):
    '''used by default'''
    return o.serialize()


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
        self.cities = ordered_cities


    def randomize(self):
        '''Randomizes the route'''
        self.cities = random.sample(self.cities, len(self.cities))
        return self


    @property
    def distance(self):
        '''The distance of this route'''
        d = 0
        for i in range(len(self.cities)-1):
            d += self.cities[i].distance_to(self.cities[i+1])
        d += self.cities[-1].distance_to(self.cities[0])
        return d


    @property
    def fitness(self):
        '''ranks the fitness of this route on scale of 0 to 1'''
        return 1 / self.distance


    @property
    def x(self):
        '''returns x as a list of city data to plot'''
        return [city.x for city in self.cities] + [self.cities[0].x]


    @property
    def y(self):
        '''returns y as a list of city data to plot'''
        return [city.y for city in self.cities] + [self.cities[0].y]


    def serialize(self):
        '''Export self as json file'''
        return self.__dict__


    def __len__(self):
        return len(self.cities)


    def __str__(self):
        return f'(d={self.distance}) (f={self.fitness})'


    def __repr__(self):
        return [c for c in self.cities]


class Population:
    def __init__(self, routes=[]):
        '''The population of possible routes the salesman can take
        param cities list: list of cities for the route'''
        #self.cities = cities
        self.routes = routes


    def randomize(self, cities, size):
        '''Randomizes (resets) the population of routes
        param size int: size of population'''
        self.routes = []
        for i in range(size):
            newroute = Route(cities).randomize()
            self.routes.append(newroute)
        return self


    def random_individual(self):
        '''Returns a random individual from the population'''
        return random.choice(self.routes)


    def rank(self):
        '''Ranks the routes within this population'''
        self.routes.sort(key=operator.attrgetter('fitness'), reverse=True)


    @property
    def ranked(self):
        '''Returns the ranked routes, but doesn't change the internal state'''
        return sorted(self.routes, key=operator.attrgetter('fitness'), reverse=True)


    @property
    def cities(self):
        '''Returns a copied list of the cities in the first route'''
        return self.routes[0].cities.copy()


    @property
    def best_individual(self):
        '''Returns the individual route with the best fitness in this population'''
        return max(self.routes, key=operator.attrgetter('fitness'))


    @property
    def max_fitness(self):
        '''Finds the maximum fitness route of the population'''
        return max(self.routes, key=operator.attrgetter('fitness')).fitness


    @property
    def min_fitness(self):
        '''Finds the minimum fitness route of the population'''
        return min(self.routes, key=operator.attrgetter('fitness')).fitness


    def serialize(self):
        '''Export self as json file'''
        return self.__dict__ #{'routes': len(self.routes), 'cities': len(self.routes[0])}


    def __repr__(self):
        return f'Pop; routes: {len(self.routes)}; cities: {len(self.routes[0])}'


class Crosser:
    def __init__(self, parents, elitesize):
        '''Crosses the parents over to create children'''
        self.parents = parents
        self.all_cities = parents.cities
        self.elitesize = elitesize


    def run(self):
        '''Runs the crossover method on all the parents'''
        children:Route = []
        for i in range(len(self.parents.routes) - self.elitesize):
            children.append(
                self.cross(self.parents.routes[i], self.parents.routes[-i-1])
                )

        children.extend(self.parents.ranked[:self.elitesize])

        return Population(children)


    def cross(self, p1, p2):
        '''Crosses parent1 with parent2
        Note that all cities must be represented'''
        
        child = []

        r1 = random.randint(0, len(p1.cities))
        r2 = random.randint(0, len(p1.cities))

        start = min(r1, r2)
        stop = max(r1, r2)

        cross_segment = p1.cities[start:stop]
        needed_cities = [city for city in self.all_cities if city not in cross_segment]

        child.extend(cross_segment)

        for i in range(len(needed_cities)):
            r = random.randint(0, len(needed_cities)-1)
            child.append(needed_cities.pop(r))

        return Route(child)


class Mutator:
    def __init__(self, children, chance):
        '''Mutates the children
        param children list: list of route children
        param chance float: chance of mutation btw 0 and 1'''
        self.children = children
        self.chance = chance

    
    def run(self):
        '''Runs the mutator'''
        for child in self.children.routes:
            self.mutate2(child)
        return Population(self.children.routes)


    def mutate(self, child):
        '''Mutates each gene in child with a chance of self.chance'''
        for i in range(len(child.cities)):
            if random.random() < self.chance:
                print('mutated')
                r = random.randint(0, len(child.cities)-1)
                child.cities[i], child.cities[r] = child.cities[r], child.cities[i]


    def mutate2(self, child):
        '''Mutates each child with a chance of self.chance'''
        if random.random() < self.chance:
            A = random.randint(0, len(child.cities)-1)
            B = random.randint(0, len(child.cities)-1)
            child.cities[A], child.cities[B] = child.cities[B], child.cities[A]

class Selector:   
    def __init__(self, population, elitesize):
        '''Selects a parent population from the population param'''
        self.population = population
        self.elitesize = elitesize


    def run(self):
        '''Runs the selector'''
        parent_routes = self.stochastic_acceptance_selection()

        # elitism
        self.population.rank()
        parent_routes.extend(self.population.routes[:self.elitesize])

        return Population(parent_routes)

                
    def tournament_selection(self):
        '''Runs the selection process for this population'''


    def stochastic_acceptance_selection(self):
        '''Selects parents based on stochastic acceptance.
        1) Randomly select individual
        2) Accept selection with probability fi/fm
        where fm = maximum population fitness'''
        max_fitness = self.population.max_fitness
        min_fitness = self.population.min_fitness
        parents = []
        complete = False
        while not complete:
            individual = self.population.random_individual()
            probality = (individual.fitness - min_fitness) / max_fitness
            if random.random() <= probality:
                parents.append(individual)
                if len(parents) == len(self.population.routes) - self.elitesize:
                    complete = True
        return parents


class GeneticAlgorithm:
    def __init__(self, cities, populationsize, elitesize, mutationrate, generations):
        self.population = Population().randomize(cities, populationsize)
        self.mutationrate = mutationrate
        self.generations = generations
        self.elitesize = elitesize
        self.best_routes = []


    def run(self):
        '''runs the genetic algorithm for the specified duration
        or perhaps until some criteria is met'''
        self.best_routes = []
        self.best_routes.append(self.population.best_individual)

        for g in range(self.generations):
            print(f'Generation {g}/{self.generations}: {1/self.population.max_fitness}')
            self.population = self.next_gen()
            self.best_routes.append(self.population.best_individual)


    def next_gen(self):
        '''using the current population, create the next generation'''
        parents = Selector(self.population, self.elitesize).run()
        children = Crosser(parents, self.elitesize).run() # TODO: ADD ELITESIZE TO THE CROSSER TOO
        children = Mutator(children, self.mutationrate).run()
        return children


    def plot(self):
        '''plots the best route data'''
        plotroutes(self.best_routes)


    def save(self, fp='gadata.json'):
        '''Export self as json file'''
        with open(fp, 'w') as f:
            json.dump(self, f, default=to_serializable)


    def serialize(self):
        return self.__dict__


def decode_ga(dct):
    '''decodes a GeneticAlgorithm object'''
    elitesize = dct['elitesize']
    generations = dct['generations']
    mutationrate = dct['mutationrate']

    best_routes = []
    for route in dct['best_routes']:
        r = Route([])
        best_routes.append(r)
        for city in route['cities']:
            r.cities.append(City(city['x'], city['y']))

    #ga = GeneticAlgorithm(cities, populationsize, elitesize, mutationrate, generations)
    return best_routes


def read_ga_file(fp):
    '''reads in a genetic algorithm saved file and plots the data'''
    with open(fp, 'r') as f:
        data = f.read()

    dct = json.loads(data)
    ga = decode_ga(dct)
    plotroutes(ga)


def run_new_ga():
    '''runs a new genetic algorithm'''
    cities = [City().randomize(0,200,0,200) for i in range(20)]
    populationsize = 10

    ga = GeneticAlgorithm(cities=cities, 
                     populationsize=100, 
                     elitesize=20, 
                     mutationrate=0.01, 
                     generations=300)
    ga.run()
    ga.save('run1_20c300g.json')
    ga.plot()


if __name__ == '__main__':

    read_ga_file('assets\\example20c300g.json')
    #run_new_ga()
    