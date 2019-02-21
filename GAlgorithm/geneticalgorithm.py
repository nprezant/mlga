
import random

from sklearn.naive_bayes import GaussianNB

from .bases import Population


def default_select(*args):
    assert 'you must assign a "select" method to the GeneticAlgorithm'


def default_crossover(*args):
    assert 'you must assign a "select" method to the GeneticAlgorithm'


def default_mutate(*args):
    assert 'you must assign a "select" method to the GeneticAlgorithm'


def cull(keep:int, *args:Population):
    '''Keeps only the best *keep* number of individuals in the populations'''
    total_pop = Population([])
    for p in args:
        total_pop.add(p)
    total_pop.rank()
    return Population(total_pop.individuals[:keep])


class GeneticAlgorithm:
    def __init__(self, initial_population, tourny_size=2, mutation_rate=0.05, f_eval_max=4000):
        self.initial_population = initial_population
        self.mutation_rate = mutation_rate
        self.tourny_size = tourny_size
        self.f_eval_max = f_eval_max
        self.reset()

        self.select = default_select
        self.crossover = default_crossover
        self.mutate = default_mutate
        

    def evolve(self, population, tourny_size, mutation_rate):
        '''Evolves the population to the next generation
        Returns: a Population of the new generation'''
        parents = self.select(population, tourny_size)
        children1 = self.crossover(parents)
        children2 = self.mutate(children1, mutation_rate)
        return children2


    def reset(self):
        '''resets the variables necessary to re-run the genetic algorithm'''
        self.f_evals = 0
        self.pop_history = []


    def run_without_ml(self):
        '''runs the genetic algorithm without machine learning'''
        self.reset()
        pop_size = len(self.initial_population.individuals)

        new_population = self.initial_population.copy()
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
            new_population = self.evolve(population, self.tourny_size, self.mutation_rate)


    def run_with_ml(self):
        '''runs the genetic algorithm with machine learning'''
        self.reset()
        self.classifier = GaussianNB()

        pop_size = len(self.initial_population.individuals)

        new_population = self.initial_population.copy()
        self.pop_history.append(new_population)

        low_dev_count = 0 # count of number of times the population had low deviation in a row
        no_improvement_count = 0 # count of generations that showed no improvement

        while self.f_evals < self.f_eval_max:
            # run objective function and keep count
            objective_calls = new_population.evaluate()
            self.f_evals += objective_calls

            # only keep the best of the old and new population
            population = cull(pop_size, self.pop_history[-1], new_population)

            # figure out if generation improved
            if (new_population.max_fitness >= self.pop_history[-1].max_fitness) and (new_population.mean_fitness >= self.pop_history[-1].mean_fitness):
                print('no improvement in this generation')
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            # add population to the history, and update the function evals it took
            population.f_evals = objective_calls
            self.pop_history.append(population)

            # train classifier with the population history
            self.train_classifier(self.pop_history)

            # evolve population. if the population has a low standard deviation, try mutation
            dev = population.get_standard_deviation()
            # print(f'Population standard deviation: {dev}')
            if dev < 5e-6 or no_improvement_count > 2:
                new_population = self.mutate(population.copy(), 1)
                # break the loop if the we have had 5 instances of low deviation
                low_dev_count += 1
                if low_dev_count > 3:
                    print(f'broken at eval = {self.f_evals}')
                    break
                print('just mutated this time')
            else:
                low_dev_count = 0
                new_population = self.evolve(population, self.tourny_size, self.mutation_rate)

            # classify
            good_pop, bad_pop = self.classify(new_population)
            new_population = good_pop
            #print(f'Classifier chose {len(good_pop.individuals)} children as good')


    def train_classifier(self, poph):
        '''Updates the classifier a list of all the prior populations given'''

        # make one giant population out of the population history list
        combined_pop = Population([])
        for p in poph:
            combined_pop.add(p)

        # make X training data in the form of [[x1,x2...y1,y2...], ...]
        x_train = [r.make_training_data() for r in combined_pop.individuals]

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