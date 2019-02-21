# -*- coding: utf-8 -*-

import json

from .tsp_bases import City, Route, random_population, select, crossover, mutate
from .tsp_plot import plot_histories

import GAlgorithm


def decode_route(dct):
    '''reads in the starter cities from a saved file'''
    cities = []
    for city in dct['_genes']:
        cities.append(City(city['x'], city['y']))
    return cities


def run_new_ga():
    '''runs a new genetic algorithm'''
    #cities = [City().randomize(0,200,0,200) for i in range(10)]
    with open('cities\\starter_cities10.txt') as f: cities_dict = json.load(f)
    cities = decode_route(cities_dict)
    init_pop = random_population(cities,100)

    ga = GAlgorithm.GeneticAlgorithm(initial_population=init_pop,
                          tourny_size=2, 
                          mutation_rate=0.05,
                          f_eval_max=3500)

    ga.select = select
    ga.crossover = crossover
    ga.mutate = mutate

    ga.run_without_ml()
    hist1 = ga.pop_history.copy()
    for i in hist1:
        i.evaluate()

    ga.run_with_ml()
    hist2 = ga.pop_history.copy()
    for i in hist2:
        i.evaluate()

    plot_histories([(hist1, 'GA'), (hist2, 'GA with ML')])


def make_cities(numcities:int):
    cities = [City().randomize(0,200,0,200) for i in range(numcities)]
    route = Route(cities)
    GAlgorithm.dump(route, f'cities\\newcity{numcities}.txt')


if __name__ == '__main__':

    run_new_ga()
    #make_cities(30)
    