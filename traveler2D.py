
import random
import operator

import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    

    def __repr__(self):
        return f'City ({self.x}, {self.y})'


class RouteWrapper:
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


    def __len__(self):
        return len(self.cities)


    def __repr__(self):
        return f'(d={self.distance}) (f={self.fitness})'


class PopulationWrapper:
    def __init__(self, routes=[]):
        '''The population of possible routes the salesman can take
        param cities list: list of cities for the route'''
        self.cities = cities
        self.routes = routes


    def randomize(self, cities, size):
        '''Randomizes (resets) the population of routes
        param size int: size of population'''
        self.routes = []
        for i in range(size):
            newroute = RouteWrapper(cities).randomize()
            self.routes.append(newroute)
        return self


    def random_individual(self):
        '''Returns a random individual from the population'''
        return random.choice(self.routes)


    def random_gene(self):
        '''Returns a random gene (city) from the city options'''
        return random.choice(self.cities)


    def rank(self):
        '''Ranks the routes within this population'''
        self.routes.sort(key=operator.attrgetter('fitness'), reverse=True)


    @property
    def ranked(self):
        '''Returns the ranked routes, but doesn't change the internal state'''
        return sorted(self.routes, key=operator.attrgetter('fitness'), reverse=True)


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
        children:RouteWrapper = []
        for i in range(len(self.parents.routes) - self.elitesize):
            children.append(
                self.cross(self.parents.routes[i], self.parents.routes[-i-1])
                )

        children.extend(self.parents.ranked[:self.elitesize])

        return PopulationWrapper(children)


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

        return RouteWrapper(child)


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
        return PopulationWrapper(self.children.routes)


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

        return PopulationWrapper(parent_routes)

                
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
        self.population = PopulationWrapper().randomize(cities, populationsize)
        self.mutationrate = mutationrate
        self.generations = generations
        self.elitesize = elitesize


    def run(self):
        '''runs the genetic algorithm for the specified duration
        or perhaps until some criteria is met'''
        best_routes = []
        best_routes.append(self.population.best_individual)

        for g in range(self.generations):
            print(f'Distance {g}: {1/self.population.max_fitness}')
            self.population = self.next_gen()
            best_routes.append(self.population.best_individual)

        plotroutes(best_routes)


    def next_gen(self):
        '''using the current population, create the next generation'''
        parents = Selector(self.population, self.elitesize).run()
        children = Crosser(parents, self.elitesize).run() # TODO: ADD ELITESIZE TO THE CROSSER TOO
        children = Mutator(children, self.mutationrate).run()
        return children


    def serialize(self):
        '''Turn self into json notation'''



def plotroutes(routes):
    '''plots a list of routes as an animation'''
    fig, (ax1, ax2) = plt.subplots(2, 1)

    title = ax1.text(0.5,0.5, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
            transform=ax1.transAxes, ha="center")

    cityMarks, = ax1.plot(routes[0].x, routes[0].y, 'b*')
    routeLine, = ax1.plot(routes[0].x, routes[0].y, 'r--')

    gen = [i for i in range(len(routes))]
    dist = [r.distance for r in routes]
    frontier, = ax2.plot(gen, dist, '-')
    frontierTrace, = ax2.plot(gen[0], dist[0], '*')

    def animate(frame, routeLine, routes, frontierTrace):
        '''updates the plot'''

        title.set_text(f'Generation: {frame}\n'
                       'Route distance: {:8.2f}\n'.format(routes[frame].distance) +
                       'Initial Distance: {:8.2f}'.format(routes[0].distance))

        routeLine.set_xdata(routes[frame].x)
        routeLine.set_ydata(routes[frame].y)

        frontierTrace.set_xdata(frame)
        frontierTrace.set_ydata(routes[frame].distance)

        return routeLine, title, frontierTrace

    ani = animation.FuncAnimation(
        fig=fig, func=animate, frames=len(routes), fargs=(routeLine, routes, frontierTrace), interval=200, blit=True)

    plt.show()




if __name__ == '__main__':

    #random.seed(90001)

    cities = [City().randomize(0,200,0,200) for i in range(20)]
    populationsize = 10

    GeneticAlgorithm(cities=cities, 
                     populationsize=100, 
                     elitesize=10, 
                     mutationrate=0.01, 
                     generations=300).run()