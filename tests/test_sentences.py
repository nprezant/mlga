
import random

from GAlgorithm import (
    GeneticAlgorithm,
    initialize_population)


def fitness(sentence, target):
    '''Computes fitness of the string compared to the base string'''
    fitness = 0
    for i,g in enumerate(sentence.genes):
        if g.value == target[i]:
            fitness += 1
    return fitness


def run():
    '''runs a new string pattern matching GA'''
    allowed_vals = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !,.'
    target = 'Hello world, I mostly work.'
    init_pop = initialize_population(500, len(target), allowed_vals)
    ga = GeneticAlgorithm(init_pop, fitness, 2, 0.05, 30000)
    ga.fitness_params = {'target':target}
    ga.run_without_ml()

    for p in ga.pop_history:
        print('Fitness={}: {}'.format(p.best_individual.fitness, p.best_individual))

if __name__ == "__main__":
    run()