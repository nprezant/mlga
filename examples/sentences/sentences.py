
import random

from GAlgorithm import (
    GeneticAlgorithm,
    initialize_population,
    fitness_plot)


def fitness(sentence, target):
    '''Computes fitness of the string compared to the base string'''
    fitness = 0
    for i,g in enumerate(sentence.genes):
        if g.value == target[i]:
            fitness += 1
    return fitness


def training_data(sentence):
    '''Turns a sentence into classifier-digestable training data'''
    return [ord(character.value) for character in sentence.genes]


def run():
    '''runs a new string pattern matching GA'''
    allowed_vals = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !,.'
    target = 'Hello world, this is hard.'
    init_pop = initialize_population(500, len(target), allowed_vals)
    ga = GeneticAlgorithm(
        init_pop, 
        fitness, 
        2, 
        0.05, 
        30000, 
        training_data_function=training_data,
        classifier_percentage=0.2
    )
    ga.fitness_params = {'target':target}

    ga.run_without_ml()
    hist1 = (ga.pop_history.copy(), 'Without ML')
    for p in ga.pop_history:
        print('Fitness={}: {}'.format(p.best_individual.fitness, p.best_individual))

    print('now WITH MACHINE LEARNING')
    ga.run_with_ml()
    hist2 = (ga.pop_history.copy(), 'With ML')
    for p in ga.pop_history:
        print('Fitness={}: {}'.format(p.best_individual.fitness, p.best_individual))

    fitness_plot([hist1, hist2], 'Sentence Fitness Plot')


if __name__ == "__main__":
    run()
