
from geneticalgorithm import GeneticAlgorithm
from ML import Classifier
from bases import City, Route, random_population
from singleplot import plotroutes
import savejson


def plot_from_file(fp):
    '''reads in a genetic algorithm saved file and plots the data'''
    dct = savejson.load(fp)
    plotroutes(dct['best_routes'])


def run_new_ga():
    '''runs a new genetic algorithm'''
    #cities = [City().randomize(0,200,0,200) for i in range(20)]
    cities = savejson.load('cities\\starter_cities10.txt')
    init_pop = random_population(cities,100)

    ga = GeneticAlgorithm(initial_population=init_pop,
                     tourny_size=2, 
                     mutation_rate=0.01,
                     f_eval_max=8000)
    ga.run_without_ml()
    #savejson.dump(ga, 'savedata\\mltest.txt')
    for i in ga.pop_history:
        i.evaluate()
    best_routes = [i.best_individual for i in ga.pop_history]
    plotroutes(ga.pop_history)


def make_cities(numcities:int):
    cities = [City().randomize(0,200,0,200) for i in range(numcities)]
    route = Route(cities)
    savejson.dump(route, f'cities\\newcity{numcities}.txt')
    plotroutes([route])


if __name__ == '__main__':

    #plot_from_file('savedata\\00_c20_p300_e20_m0.01_g400_mCART.txt')
    run_new_ga()
    #make_cities(30)
    