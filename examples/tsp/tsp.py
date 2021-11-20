# -*- coding: utf-8 -*-

from pathlib import Path

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from mlga import (
    GeneticAlgorithm,
    fitness_plot,
    SaveLocation,
    Algorithm
)

from .population import (
    City,
    Route, 
    compute_fitness,
    random_population,
    make_training_data
)

from .evolution import (
    crossover, 
    mutate
)
    
from .plot import plot_histories
from .io import read_cities


DEFAULT_SAVES = SaveLocation(
    Path().cwd() / Path('data'),
    'Default',
)

    
def make_initial_population(cities_fp, size):
    cities = read_cities(cities_fp)
    return random_population(cities, size)

GA_ARGS = {
    'initial_population': make_initial_population('cities/10cities.txt', 100),
    'fitness_function': compute_fitness,
    'training_data_function': make_training_data,
    'tourny_size': 2, 
    'mutation_rate': 0.05,
    'f_eval_max': 50000,
    'classifier_percentage': 0.25,
    'classifier_class': KNeighborsClassifier,
    'crossover_fn': crossover,
    'mutate_fn': mutate
}

def run_comparison():
    '''runs a new genetic algorithm n number of times'''
    #cities = [City().randomize(0,200,0,200) for i in range(10)]

    ga = GeneticAlgorithm(**GA_ARGS)

    ga.run()
    hist1 = ga.pop_history.copy()

    ga.run_with_ml()
    hist2 = ga.pop_history.copy()

    fitness_plot([(hist1, 'GA'), (hist2, 'GA with ML')], 'Travelling Salesman Problem')
    plot_histories([(hist1, 'GA'), (hist2, 'GA with ML')])

def run(iterations=1, saves=DEFAULT_SAVES,  algorithm=Algorithm.STANDARD, **kwargs):

    # update the GA input parameters
    GA_ARGS.update(kwargs)

    # initialize the GA
    ga = GeneticAlgorithm(**GA_ARGS)
    
    # ensure that the save directory exists
    saves.base_folder.mkdir(parents=True, exist_ok=True)

    # write out the GA parameters
    ga.write_params(saves.params_fp())

    # iterate the GA
    for n in range(iterations):

        # run the GA, depending on algorithm option
        if algorithm == Algorithm.STANDARD:
            ga.run()
        elif algorithm == Algorithm.RANDOM:
            ga.run_random()
        elif algorithm == Algorithm.ML:
            ga.classifer_vars_fp = saves.performance_fp(n)
            ga.run_with_ml()
        else:
            raise KeyError()

        # save the population history
        ga.pop_history.to_csv(saves.run_fp(n))

        # save the best individual in this run
        ga.pop_history[-1].best_individual.to_csv(saves.best_fp(n))
