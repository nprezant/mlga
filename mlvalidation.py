
from operator import itemgetter



from geneticalgorithm import GeneticAlgorithm
import savejson

models = ['LR', 'LDA', 'KNN', 'CART', 'None']
#models = ['None', 'None', 'None', 'None']

cities = savejson.load('cities\\starter_cities20.txt')

s = ''

for i,model in enumerate(models):

    print(f'Running {model} model...')

    populationsize = 300
    elitesize = 60
    touramentsize = 2
    mutationrate = 0.01
    generations = 500

    ga = GeneticAlgorithm(cities,
                          populationsize,
                          elitesize,
                          touramentsize,
                          mutationrate,
                          generations,
                          model)
    ga.run()

    # index of route with minimum distance
    dist = [r.distance for r in ga.best_routes]
    imin = min(enumerate(dist), key=itemgetter(1))[0]

    log = f'\n{model}: Best Route (#{imin}): {ga.best_routes[imin].distance}'
    with open('run_log.txt', 'a') as f:
        f.write(log)

    s = s + log
    savepath = f'savedata\\04_c{len(cities)}_p{populationsize}_e{elitesize}_m{mutationrate}_g{generations}_m{model}_i{i}.txt'
    savejson.dump(ga, savepath)

print(s)