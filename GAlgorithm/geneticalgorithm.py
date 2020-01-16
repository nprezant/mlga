
import random
from pathlib import Path

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from .population import Population, Objective, PopulationHistory
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
        classifier_percentage=0.4,
        select_fn=tournament_selection,
        crossover_fn=order_independent_crossover,
        mutate_fn=gene_based_mutation,
        tracked_variables_fp=Path('tracked_variables.txt')
    ):
        self.initial_population = initial_population

        self.mutation_rate = mutation_rate
        self.tourny_size = tourny_size
        self.f_eval_max = f_eval_max

        self.fitness_function = fitness_function
        self.fitness_params = fitness_params

        self.training_data_function = training_data_function
        self.classifier_percentage = classifier_percentage
        self.classifer_vars_fp = tracked_variables_fp

        self.select = select_fn
        self.crossover = crossover_fn
        self.mutate = mutate_fn

        self.reset()
        
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
        self.pop_history = PopulationHistory()
        self.classifer_var_tracker = {
            'FunctionEvaluation': [],
            'ClassifiedGood': [],
            'ClassifiedBad': [],
            'ClassifiedGoodActuallyGood': [],
            'ClassifiedGoodActuallyBad': [],
            'ClassifiedBadActuallyBad': [],
            'ClassifiedBadActuallyGood': [],
            'GoodPredictorPercentage': [],
            'BadPredictorPercentage': [],
        }

    def run_random(self):
        '''generates random population with the "initialize" function

        culls new populations to ensure the fitness does not move backwards
        '''

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

            # create new population
            new_population = population.copy()
            for p in new_population.individuals:
                p.randomize()

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

            # progress update
            # print(
            #     f'Fn evals: {self.f_evals}/{self.f_eval_max} '
            #     f'Classified {len(good_pop)} children as good.'
            # )

            self.record_tracker_data(good_pop.copy(), bad_pop.copy(), new_population.copy())
            new_population = good_pop

        self.save_tracked_variables(self.classifer_vars_fp)

    def record_tracker_data(self, classified_good_pop, classified_bad_pop, total_pop):

        # record data
        self.classifer_var_tracker['FunctionEvaluation'].append(self.f_evals)
        self.classifer_var_tracker['ClassifiedGood'].append(len(classified_good_pop))
        self.classifer_var_tracker['ClassifiedBad'].append(len(classified_bad_pop))
        
        # evaluate all of the populations
        classified_good_pop.evaluate(self.fitness_function, self.fitness_params)
        classified_bad_pop.evaluate(self.fitness_function, self.fitness_params)
        total_pop.evaluate(self.fitness_function, self.fitness_params)

        # determine performance of classifying "good"
        if len(classified_good_pop) == 0:
            classified_good_actually_good = 0
            classified_good_actually_bad = 0
        else:
            min_fitness = total_pop.min_fitness
            max_fitness = total_pop.max_fitness

            delta = max_fitness - min_fitness
            perc = self.classifier_percentage

            if Population.objective_type == Objective.MAXIMIZE:
                cutoff = max_fitness - (delta * perc)
                classified_good_actually_good_pop = [i for i in classified_good_pop.individuals if i.fitness > cutoff]
                classified_good_actually_bad_pop = [i for i in classified_good_pop.individuals if i.fitness < cutoff]
            else:
                cutoff = min_fitness + (delta * perc)
                classified_good_actually_good_pop = [i for i in classified_good_pop.individuals if i.fitness < cutoff]
                classified_good_actually_bad_pop = [i for i in classified_good_pop.individuals if i.fitness > cutoff]

            classified_good_actually_good = len(classified_good_actually_good_pop)
            classified_good_actually_bad = len(classified_good_actually_bad_pop)

        self.classifer_var_tracker['ClassifiedGoodActuallyGood'].append(classified_good_actually_good)
        self.classifer_var_tracker['ClassifiedGoodActuallyBad'].append(classified_good_actually_bad)

        # determine performance of classifying "bad"
        if len(classified_bad_pop) == 0:
            classified_bad_actually_good = 0
            classified_bad_actually_bad = 0
        else:
            min_fitness = total_pop.min_fitness
            max_fitness = total_pop.max_fitness

            delta = max_fitness - min_fitness
            perc = self.classifier_percentage

            if Population.objective_type == Objective.MAXIMIZE:
                cutoff = max_fitness - (delta * perc)
                classified_bad_actually_good_pop = [i for i in classified_bad_pop.individuals if i.fitness > cutoff]
                classified_bad_actually_bad_pop = [i for i in classified_bad_pop.individuals if i.fitness < cutoff]
            else:
                cutoff = min_fitness + (delta * perc)
                classified_bad_actually_good_pop = [i for i in classified_bad_pop.individuals if i.fitness < cutoff]
                classified_bad_actually_bad_pop = [i for i in classified_bad_pop.individuals if i.fitness > cutoff]

            classified_bad_actually_good = len(classified_bad_actually_good_pop)
            classified_bad_actually_bad = len(classified_bad_actually_bad_pop)
        
        self.classifer_var_tracker['ClassifiedBadActuallyBad'].append(classified_bad_actually_bad)
        self.classifer_var_tracker['ClassifiedBadActuallyGood'].append(classified_bad_actually_good)

        # how accurate is the classifier?
        try:
            self.classifer_var_tracker['GoodPredictorPercentage'].append(classified_good_actually_good / len(classified_good_pop))
        except ZeroDivisionError:
            if classified_bad_actually_good == 0:
                self.classifer_var_tracker['GoodPredictorPercentage'].append(1)
            else:
                self.classifer_var_tracker['GoodPredictorPercentage'].append(0)

        try:
            self.classifer_var_tracker['BadPredictorPercentage'].append(classified_bad_actually_bad / len(classified_bad_pop))
        except ZeroDivisionError:
            if classified_good_actually_bad == 0:
                self.classifer_var_tracker['BadPredictorPercentage'].append(1)
            else:
                self.classifer_var_tracker['BadPredictorPercentage'].append(0)

    def save_tracked_variables(self, fp: Path):
        
        # classifier accuracy indicators
        good_percentages = self.classifer_var_tracker['GoodPredictorPercentage']
        good_accuracy = sum(good_percentages) / len(good_percentages)

        bad_percentages = self.classifer_var_tracker['BadPredictorPercentage']
        bad_accuracy = sum(bad_percentages) / len(bad_percentages)

        accuracy_summary = (
            'Classifier predicts good children with an accuracy of '
            f'{round(good_accuracy*100, 1)}%, and bad children with '
            f'an accuracy of {round(bad_accuracy*100, 1)}%\r'
        )

        if not fp.exists():
            fp.touch()
        
        with open(fp, 'w') as f:

            print(accuracy_summary)
            f.write(accuracy_summary)

            # headers
            headers = []
            for header in self.classifer_var_tracker.keys():
                headers.append(header)
                f.write(f'{header}\t')

            # add mean fitness header
            f.write('Mean Fitness\r')

            # values
            for row in range(len(self.classifer_var_tracker[headers[0]])):
                for header in headers:
                    item = self.classifer_var_tracker[header]
                    f.write(f'{item[row]}\t')

                f.write(f'{self.pop_history[row].mean_fitness}\r')

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
