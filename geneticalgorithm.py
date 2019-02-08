
import random

from sklearn.naive_bayes import GaussianNB

from bases import City, Route, Population


def crossover(parents):
    '''Crosses the parents over to create children'''
    all_cities = parents.genes

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

    for i in range(len(needed_cities)):
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


def mutate_1(child):
    '''Mutates each gene in child with a chance of self.chance'''
    for i in range(len(child.genes)):
        if random.random() < self.chance:
            r = random.randint(0, len(child.genes)-1)
            child.genes[i], child.genes[r] = child.genes[r], child.genes[i]
            child._fitness = None


def mutate_2(child, chance):
    '''Mutates each child with a chance of self.chance'''
    if random.random() < chance:
        A = random.randint(0, len(child.genes)-1)
        B = random.randint(0, len(child.genes)-1)
        child.genes[A], child.genes[B] = child.genes[B], child.genes[A]
        child._fitness = None


def select(population, tourny_size:int=2):
    '''Selects a parent population from the population
    Uses a tournament to select parents'''
    parents = []
    for i in population.individuals:
        winner = population.random_individual()
        for i in range(1, tourny_size):
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


def evolve(population, tourny_size, mutation_rate):
    '''Evolves the population to the next generation
    Returns: a Population of the new generation'''
    parents = select(population, tourny_size)
    children1 = crossover(parents)
    children2 = mutate(children1, mutation_rate)
    return children2


def cull(keep:int, *args:Population):
    '''Keeps only the best *keep* number of individuals in the populations'''
    total_pop = Population([])
    for p in args:
        total_pop.add(p)
    total_pop.rank()
    return Population(total_pop.individuals[:keep])


class GeneticAlgorithm:
    def __init__(self, initial_population, tourny_size, mutation_rate, f_eval_max):
        self.initial_population = initial_population
        self.mutation_rate = mutation_rate
        self.tourny_size = tourny_size
        self.f_eval_max = f_eval_max
        self.reset()


    def reset(self):
        '''resets the variables necessary to re-run the genetic algorithm'''
        self.f_evals = 0
        self.pop_history = []


    def run_without_ml(self):
        '''runs the genetic algorithm without machine learning'''
        self.reset()
        pop_size = len(self.initial_population.individuals)

        new_population = self.initial_population
        self.pop_history.append(new_population)

        while self.f_evals < self.f_eval_max:
            # run objective function and keep count
            objective_calls = new_population.evaluate()
            self.f_evals += objective_calls

            # only keep the best of the old and new population
            population = cull(pop_size, self.pop_history[-1], new_population)

            # add population to the history, and update the function evals it took
            population.f_evals = objective_calls
            self.pop_history.append(population)

            # evolve population
            new_population = evolve(population, self.tourny_size, self.mutation_rate)


    def run_with_ml(self):
        '''runs the genetic algorithm with machine learning'''
        self.reset()
        self.classifier = GaussianNB()

        pop_size = len(self.initial_population.individuals)

        new_population = self.initial_population
        self.pop_history.append(new_population)

        while self.f_evals < self.f_eval_max:
            # run objective function and keep count
            objective_calls = new_population.evaluate()
            self.f_evals += objective_calls

            # only keep the best of the old and new population
            population = cull(pop_size, self.pop_history[-1], new_population)

            # add population to the history, and update the function evals it took
            population.f_evals = objective_calls
            self.pop_history.append(population)

            # train classifier with the population history
            self.train_classifier(self.pop_history)

            # evolve population
            new_population = evolve(population, self.tourny_size, self.mutation_rate)

            # classify
            good_pop, bad_pop = self.classify(new_population)
            new_population = good_pop
            print(f'Classifier chose {len(good_pop.individuals)} children as good')


    def train_classifier(self, poph):
        '''Updates the classifier a list of all the prior populations given'''

        # make one giant population out of the population history list
        combined_pop = Population([])
        for p in poph:
            combined_pop.add(p)

        # make X training data in the form of [[x1,x2...y1,y2...], ...]
        x_train = [r.x + r.y for r in combined_pop.individuals]

        # determine cutoff value for a "good" individual
        min_fitness = combined_pop.min_fitness
        max_fitness = combined_pop.max_fitness
        delta = max_fitness - min_fitness
        cutoff = max_fitness - (delta * 0.5) # only keep the top 20%

        # make Y training data in the form of ['good', 'bad', 'bad', ...]
        y_train = ['good' if i.fitness > cutoff else 'bad' for i in combined_pop.individuals]

        self.classifier.fit(x_train, y_train)


    def classify(self, pop):
        '''Predicts whether individivuals in the population will
        be good or bad, then returns them'''
        x_data = [c.x + c.y for c in pop.individuals]
        predictions = self.classifier.predict(x_data)
        good_indivs = [pop.individuals[i] for i,p in enumerate(predictions) if p == 'good']
        bad_indivs  = [pop.individuals[i] for i,p in enumerate(predictions) if p == 'bad']
        good_pop = Population(good_indivs)
        bad_pop = Population(bad_indivs)
        return good_pop, bad_pop


    def serialize(self):
        d = dict(__GeneticAlgorithm__=True)
        d.update(self.__dict__)
        if self.classifier is None: d['classifier'] = 'None'
        return d