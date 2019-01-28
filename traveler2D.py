
from geneticalgorithm import GeneticAlgorithm
from ML import Classifier
from baseindividuals import City, Route
from singleplot import plotroutes
import savejson


def plot_from_file(fp):
    '''reads in a genetic algorithm saved file and plots the data'''
    dct = savejson.load(fp)
    plotroutes(dct['best_routes'])


def run_new_ga():
    '''runs a new genetic algorithm'''
    #cities = [City().randomize(0,200,0,200) for i in range(20)]
    cities = savejson.load('cities\\starter_cities20.txt')

    ga = GeneticAlgorithm(cities=cities, 
                     populationsize=100, 
                     elitesize=20, 
                     tournamentsize=2, 
                     mutationrate=0.01, 
                     generations=500,
                     mlmodel='KNN')
    ga.run()
    savejson.dump(ga, 'savedata\\mltest.txt')
    plotroutes(ga.best_routes)


def make_cities(numcities:int):
    cities = [City().randomize(0,200,0,200) for i in range(numcities)]
    route = Route(cities)
    savejson.dump(route, f'cities\\newcity{numcities}.txt')
    plotroutes([route])


if __name__ == '__main__':

    #plot_from_file('savedata\\00_c20_p300_e20_m0.01_g400_mCART.txt')
    run_new_ga()
    #make_cities(30)
    