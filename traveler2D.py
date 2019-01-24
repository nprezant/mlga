
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
        '''Export self as json file'''
        return self.__dict__


    def __len__(self):
        return len(self.genes)


    def __str__(self):
        return f'(d={self.distance}) (f={self.fitness})'


    def __repr__(self):
        return [c for c in self.genes]


class Population:
    def __init__(self, routes:list=[]):
        '''The population of possible routes the salesman can take
        param cities list: list of cities for the route'''
        self.individuals = routes


    def randomize(self, cities, size):
        '''Randomizes (resets) the population of routes
        param size int: size of population'''
        self.individuals = []
        for i in range(size):
            newroute = Route(cities).randomize()
            self.individuals.append(newroute)
        return self


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


    def serialize(self):
        '''Export self as json file'''
        return self.__dict__ #{'routes': len(self.routes), 'cities': len(self.routes[0])}


    def __repr__(self):
        return f'Pop; routes: {len(self.individuals)}; cities: {len(self.individuals[0])}'


class Crosser:
    def __init__(self, parents, elitesize):
        '''Crosses the parents over to create children'''
        self.parents = parents
        self.all_cities = parents.genes
        self.elitesize = elitesize


    def run(self):
        '''Runs the crossover method on all the parents'''
        children:Route = []
        for i in range(len(self.parents.individuals) - self.elitesize):
            children.append(
                self.cross(self.parents.individuals[i], self.parents.individuals[-i-1])
                )

        children.extend(self.parents.ranked[:self.elitesize])

        return Population(children)


    def cross(self, p1, p2):
        '''Crosses parent1 with parent2
        Note that all cities must be represented'''
        
        child = []

        r1 = random.randint(0, len(p1.genes))
        r2 = random.randint(0, len(p1.genes))

        start = min(r1, r2)
        stop = max(r1, r2)

        cross_segment = p1.genes[start:stop]
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
        for child in self.children.individuals:
            self.mutate2(child)
        return Population(self.children.individuals)


    def mutate(self, child):
        '''Mutates each gene in child with a chance of self.chance'''
        for i in range(len(child.genes)):
            if random.random() < self.chance:
                r = random.randint(0, len(child.genes)-1)
                child.genes[i], child.genes[r] = child.genes[r], child.genes[i]


    def mutate2(self, child):
        '''Mutates each child with a chance of self.chance'''
        if random.random() < self.chance:
            A = random.randint(0, len(child.genes)-1)
            B = random.randint(0, len(child.genes)-1)
            child.genes[A], child.genes[B] = child.genes[B], child.genes[A]
            child._fitness = None


class Selector:   
    def __init__(self, population, elitesize:int, tsize:int=2):
        '''Selects a parent population from the population param'''
        self.population = population
        self.elitesize = elitesize
        self.tsize = tsize


    def run(self):
        '''Runs the selector'''
        #parent_individuals = self.stochastic_acceptance_selection()
        parent_individuals = self.tournament_selection()

        # elitism
        self.population.rank()
        parent_individuals.extend(self.population.individuals[:self.elitesize])

        return Population(parent_individuals)

                
    def tournament_selection(self):
        '''Runs the selection process for this population'''
        parents = []
        for i in range(len(self.population.individuals) - self.elitesize):
            winner = self.population.random_individual()
            for i in range(1, self.tsize):
                competitor = self.population.random_individual()
                if competitor.fitness > winner.fitness:
                    winner = competitor
            parents.append(winner)
        return parents


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
                if len(parents) == len(self.population.individuals) - self.elitesize:
                    complete = True
        return parents


class GeneticAlgorithm:
    def __init__(self, cities, populationsize, elitesize, tsize, mutationrate, generations):
        self.population = Population().randomize(cities, populationsize)
        self.mutationrate = mutationrate
        self.generations = generations
        self.elitesize = elitesize
        self.tsize = tsize
        self.best_routes = []


    def run(self):
        '''runs the genetic algorithm for the specified duration
        or perhaps until some criteria is met'''
        self.best_routes = []
        self.best_routes.append(self.population.best_individual)

        for g in range(self.generations):
            print(f'Generation {g}/{self.generations}: {1/self.population.best_individual.fitness}')
            self.population = self.next_gen()
            self.best_routes.append(self.population.best_individual)


    def next_gen(self):
        '''using the current population, create the next generation'''
        parents = Selector(self.population, self.elitesize, self.tsize).run()
        children = Crosser(parents, self.elitesize).run() # TODO: ADD ELITESIZE TO THE CROSSER TOO
        children = Mutator(children, self.mutationrate).run()
        return children


    def plot(self):
        '''plots the best route data'''
        plotroutes(self.best_routes)


    def save(self, fp='data.json'):
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
        for city in route['_genes']:
            r.genes.append(City(city['x'], city['y']))

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
                     tsize=2, 
                     mutationrate=0.01, 
                     generations=300)
    ga.run()
    ga.save('savedata\\run6_tournamentselection.json')
    ga.plot()


if __name__ == '__main__':

    #read_ga_file('savedata\\run3_tournamentselection.json')
    run_new_ga()
    