# -*- coding: utf-8 -*-

import json

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


def decode_route(dct):
    '''reads in the starter cities from a saved file'''
    cities = []
    for city in dct['_genes']:
        cities.append(City(city['x'], city['y']))
    return cities


def run():
    '''runs a new genetic algorithm'''
    #cities = [City().randomize(0,200,0,200) for i in range(10)]
    with open('cities\\starter_cities10.txt') as f: cities_dict = json.load(f)
    cities = decode_route(cities_dict)
    init_pop = random_population(cities,100)

    ga = GeneticAlgorithm(
        initial_population=init_pop,
        fitness_function=compute_fitness,
        training_data_function = make_training_data,
        tourny_size=2, 
        mutation_rate=0.05,
        f_eval_max=3500)

    #ga.select = select
    ga.crossover = crossover
    ga.mutate = mutate

    ga.run_without_ml()
    hist1 = ga.pop_history.copy()
    # for i in hist1:
    #     i.evaluate(compute_fitness)

    ga.run_with_ml()
    hist2 = ga.pop_history.copy()
    # for i in hist2:
    #     i.evaluate(compute_fitness)

    fitness_plot([(hist1, 'GA'), (hist2, 'GA with ML')])
    plot_histories([(hist1, 'GA'), (hist2, 'GA with ML')])


def make_cities(numcities:int):
    cities = [City().randomize(0,200,0,200) for i in range(numcities)]
    route = Route(cities)
    dump(route, f'cities\\newcity{numcities}.txt')


if __name__ == '__main__':

    run()
    #make_cities(30)
    