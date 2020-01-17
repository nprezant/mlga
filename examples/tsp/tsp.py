# -*- coding: utf-8 -*-

import json
from pathlib import Path

from GAlgorithm import (
    GeneticAlgorithm,
    dump,
    fitness_plot)

from .population import (
    City,
    Route, 
    compute_fitness,
    random_population,
    make_training_data)

from .evolution import (
    crossover, 
    mutate)
    
from .plot import plot_histories

DEFAULT_SAVE_DIRECTORY = 'data'

def decode_route(dct):
    '''reads in the starter cities from a saved file'''
    cities = []
    for city in dct['_genes']:
        cities.append(City(city['x'], city['y']))
    return cities

def read_cities(fp):
    with open(fp) as f:
        cities_dict = json.load(f)
    cities = decode_route(cities_dict)
    return random_population(cities,100)

GA_ARGS = {
    'initial_population': read_cities('cities\\starter_cities10.txt'),
    'fitness_function': compute_fitness,
    'training_data_function' :  make_training_data,
    'tourny_size': 2, 
    'mutation_rate': 0.05,
    'f_eval_max': 5000,
    'classifier_percentage': 0.25,
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

def run_standard(num_iterations=1, save_directory=DEFAULT_SAVE_DIRECTORY):
   
    ga = GeneticAlgorithm(**GA_ARGS)
    
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    ga.write_params(Path(save_directory) / Path('_params.txt'))

    for n in range(num_iterations):

        fp = Path(save_directory) / Path(f'StandardRun{n}.txt')
        fp.touch()

        ga.run()
        ga.pop_history.to_csv(fp)

def run_ml_mod(num_iterations=1, save_directory=DEFAULT_SAVE_DIRECTORY):
    
    ga = GeneticAlgorithm(**GA_ARGS)

    Path(save_directory).mkdir(parents=True, exist_ok=True)

    ga.write_params(Path(save_directory) / Path('_params.txt'))
    
    for n in range(num_iterations):

        plot_fp = Path(save_directory) / Path(f'MLRun{n}.txt')
        plot_fp.touch()

        classifier_vars_fp = Path(save_directory) / Path(f'MLClassifierVarsRun{n}.txt')
        classifier_vars_fp.touch()

        ga.classifer_vars_fp = classifier_vars_fp
        ga.run_with_ml()
        ga.pop_history.to_csv(plot_fp)

def run_random(num_iterations=1, save_directory=DEFAULT_SAVE_DIRECTORY):
    
    ga = GeneticAlgorithm(**GA_ARGS)

    Path(save_directory).mkdir(parents=True, exist_ok=True)

    ga.write_params(Path(save_directory) / Path('_params.txt'))
    
    for n in range(num_iterations):

        plot_fp = Path(save_directory) / Path(f'RandomRun{n}.txt')
        plot_fp.touch()

        ga.run_random()
        ga.pop_history.to_csv(plot_fp)

def make_cities(numcities: int):
    cities = [City().randomize(0,200,0,200) for i in range(numcities)]
    route = Route(cities)
    return route

def write_cities(numcities: int):
    route = make_cities(numcities)
    dump(route, f'cities\\newcity{numcities}.txt')


if __name__ == '__main__':

    run_comparison()
    #make_cities(30)
