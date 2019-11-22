
import random

from sklearn.naive_bayes import GaussianNB

from .population import Population, Objective
from .evolution import (
    order_independent_crossover, 
    tournament_selection, 
    gene_based_mutation,
    cull
)


class GeneticAlgorithm:

    def __init__(
        self, 
        initial_population, 
        fitness_function, 
        tourny_size=2, 
        mutation_rate=0.05, 
        f_eval_max=4000, 
        fitness_params={}, 
        training_data_function=None, 
        classifier_percentage=0.4
    ):
        self.initial_population = initial_population
        self.mutation_rate = mutation_rate
        self.tourny_size = tourny_size
        self.f_eval_max = f_eval_max
        self.training_data_function = training_data_function
        self.fitness_function = fitness_function
        self.fitness_params = fitness_params
        self.classifier_percentage = classifier_percentage
        self.reset()

        self.select = tournament_selection
        self.crossover = order_independent_crossover
        self.mutate = gene_based_mutation
        
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

    # def run_random(self):
    #     '''generates random population with the "initialize" function

    #     culls new populations to ensure the fitness does not move backwards'''

    #     self.reset()
    #     pop_size = len(self.initial_population.individuals)

    #     new_population = self.initial_population.copy()
    #     self.pop_history.append(new_population)

    #     while self.f_evals < self.f_eval_max:
    #         # run objective function and keep count
    #         objective_calls = new_population.evaluate(self.fitness_function, self.fitness_params)
    #         self.f_evals += objective_calls

    #         # only keep the best of the old and new population
    #         population = cull(pop_size, self.pop_history[-1], new_population)

    #         # add population to the history, and update the function evals it took
    #         population.f_evals = objective_calls
    #         self.pop_history.append(population)

    #         # evolve population
    #         new_population = self.evolve(population, self.tourny_size, self.mutation_rate)

    def run(self):
        '''runs the genetic algorithm without machine learning'''

        self.reset()
        pop_size = len(self.initial_population.individuals)

        new_population = self.initial_population.copy()
        self.pop_history.append(new_population)

        while self.f_evals < self.f_eval_max:
            # run objective function and keep count
            objective_calls = new_population.evaluate(self.fitness_function, self.fitness_params)
            self.f_evals += objective_calls

            # only keep the best of the old and new population
            population = cull(pop_size, self.pop_history[-1], new_population)

            # add population to the history, and update the function evals it took
            population.f_evals = objective_calls
            self.pop_history.append(population)

            # evolve population
            new_population = self.evolve(population, self.tourny_size, self.mutation_rate)

    def test_convergance(self):
        '''Checks some convergance criteria'''

        # figure out if generation improved
        if self.pop_history[-1].better_than(self.pop_history[-2]):
            improved = True
        else:
            improved = False

        # standard deviation of population
        dev = self.pop_history[-1].get_standard_deviation()
        if dev < 0.0000001:
            low_dev = True
        else:
            low_dev = False

        return improved, low_dev

    def run_with_ml(self):
        '''runs the genetic algorithm with machine learning'''
        if self.training_data_function is None:
            raise Exception('Training data function is not defined!')

        self.reset()
        self.classifier = GaussianNB()

        pop_size = len(self.initial_population.individuals)

        new_population = self.initial_population.copy()
        self.pop_history.append(new_population)

        low_dev_count = 0 # count of number of times the population had low deviation in a row
        no_improvement_count = 0 # count of generations that showed no improvement

        while self.f_evals < self.f_eval_max:
            # run objective function and keep count
            objective_calls = new_population.evaluate(self.fitness_function, self.fitness_params)
            self.f_evals += objective_calls

            # only keep the best of the old and new population
            population = cull(pop_size, self.pop_history[-1], new_population)

            # add population to the history, and update the function evals it took
            population.f_evals = objective_calls
            self.pop_history.append(population)

            # train classifier with the population history
            self.train_classifier(self.pop_history)

            # check population for convergance
            improved, low_dev = self.test_convergance()
            if low_dev:
                low_dev_count += 1
            else:
                low_dev = 0

            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            # evolve population. if the population has a low standard deviation, try mutation
            
            if low_dev_count > 1 or no_improvement_count > 2:

                new_population = self.mutate(population.copy(), 1)
                print(f'just mutated this time. no_improvement={no_improvement_count}, low_dev={low_dev_count}')

                # break the loop if the we have had 5 instances of low deviation
                if low_dev_count > 2 or no_improvement_count > 4:
                    print(f'broken at eval = {self.f_evals}')
                    break
                                    
            else:
                new_population = self.evolve(population, self.tourny_size, self.mutation_rate)

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
        x_train = [self.training_data_function(x) for x in combined_pop.individuals]

        # determine cutoff value for a "good" individual
        min_fitness = combined_pop.min_fitness
        max_fitness = combined_pop.max_fitness
        delta = max_fitness - min_fitness

        # set cutoff based on whether we are max-ing or min-ing
        # make Y training data in the form of ['good', 'bad', 'bad', ...] # use same culling function?
        perc = self.classifier_percentage
        if Population.objective_type == Objective.MAXIMIZE:
            cutoff = max_fitness - (delta * perc) # only keep the top xx%
            y_train = ['good' if i.fitness > cutoff else 'bad' for i in combined_pop.individuals]
        else:
            cutoff = min_fitness + (delta * perc) # only keep the top xx%
            y_train = ['good' if i.fitness < cutoff else 'bad' for i in combined_pop.individuals]

        self.classifier.fit(x_train, y_train)

    def classify(self, pop):
        '''Predicts whether individivuals in the population will
        be good or bad, then returns them'''
        x_data = [self.training_data_function(x) for x in pop.individuals]
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
