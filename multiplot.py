
import sys
from operator import itemgetter

import matplotlib.pyplot as plt

import savejson


def multiplotroutes(runs):
    '''Plots an overlay of mutliple genetic alg runs
    runs must be a save object dictionaries'''
    fig = plt.figure()
    ax1 = fig.add_axes([0.10, 0.10, 0.80, 0.80]) # route plot

    ax1.set_title('Comparision of ML Algorithms on Travelling Salesman Problem')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Total Distance')

    # add each run to the plot
    for run in runs:
        # frontier plot data
        gens = [i for i in range(len(run['best_routes']))]
        dist = [r.distance for r in run['best_routes']]

        fLine, = ax1.plot(gens, dist, ':', label=run['classifier'])

    ax1.legend()
    plt.show()


if __name__ == '__main__':
    dir = sys.argv[1]
    pattern = sys.argv[2]
    runs = savejson.load_from_files(dir, pattern)
    multiplotroutes(runs)