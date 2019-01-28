
import random

from baseindividuals import City, Route, Population
from ML import Classifier


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
        for i in range(self.population.size - self.elitesize):
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
    def __init__(self, cities, populationsize, elitesize, tournamentsize, mutationrate, generations, mlmodel=None):
        self.population = Population([]).randomize(cities, populationsize)
        self.poph = Population([]) # population history
        self.mutationrate = mutationrate
        self.generations = generations
        self.elitesize = elitesize
        self.tournamentsize = tournamentsize

        self.best_routes = []
        if mlmodel is None or mlmodel == 'None':
            self.classifier = None
        else:
            self.classifier = Classifier(mlmodel)


    def run(self):
        '''runs the genetic algorithm for the specified duration
        or perhaps until some criteria is met'''
        self.best_routes = []
        self.best_routes.append(self.population.best_individual)

        for g in range(self.generations):
            print(f'Generation {g}/{self.generations}: {1/self.population.best_individual.fitness}')
            if not self.classifier is None: self.update_classifier(self.population)
            self.population = self.next_gen()
            self.best_routes.append(self.population.best_individual)
            if not self.classifier is None:
                good_pop, bad_pop = self.classify_and_filter(self.population)
                self.population = good_pop
            


    def next_gen(self):
        '''using the current population, create the next generation'''
        parents = Selector(self.population, self.elitesize, self.tournamentsize).run()
        children = Crosser(parents, self.elitesize).run()
        children = Mutator(children, self.mutationrate).run()
        return children


    def update_classifier(self, pop:Population):
        '''Updates the classifier with the population given'''

        # add the new population to the population history
        #self.poph.add(pop)

        x_train = [c.x + c.y for c in pop.individuals]

        mean_fitness = pop.mean_fitness
        y_train = ['good' if i.fitness > mean_fitness else 'bad' for i in pop.individuals]

        self.classifier.training_data.replace_x(x_train)
        self.classifier.training_data.replace_y(y_train)

        self.classifier.re_train()


    def classify_and_filter(self, pop):
        '''Predicts whether individivuals in the population will
        be good or bad, then returns them'''
        old_pop_size = len(pop.individuals)
        #x_data = [c.x + c.y for c in pop.individuals]
        #predictions = self.classifier.predict(x_data)
        #good_indivs = [pop.individuals[i] for i,p in enumerate(predictions) if p == 'good']
        #bad_indivs  = [pop.individuals[i] for i,p in enumerate(predictions) if p == 'bad']
        #good_pop = Population(good_indivs, old_pop_size)
        #bad_pop = Population(bad_indivs, old_pop_size)
        #if len(good_pop.individuals) < 5:
        if True:
            #print('Classifier did not do a good job')
            mean_fitness = pop.mean_fitness
            actual_good = [i for i in pop.individuals if i.fitness > mean_fitness]
            actual_good_pop = Population(actual_good, old_pop_size)
            bad_pop = actual_good_pop
            return actual_good_pop, bad_pop
        return good_pop, bad_pop


    def serialize(self):
        d = dict(__GeneticAlgorithm__=True)
        d.update(self.__dict__)
        if self.classifier is None: d['classifier'] = 'None'
        return d