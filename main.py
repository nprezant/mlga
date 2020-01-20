
from pathlib import Path

import matplotlib.pyplot as plt

import examples

from GAlgorithm import (
    SaveLocation,
    SaveData,
    Algorithm
)

n = 1
folder = Path().cwd() / Path(f'data{n}')

rnd_saves = SaveLocation(folder, 'Random')
std_saves = SaveLocation(folder, 'Standard')
ml_saves = SaveLocation(folder, 'ML')

run = False
plot = True

# generate GA run data
if run:
    examples.run_tsp(n, rnd_saves, algorithm=Algorithm.RANDOM)
    examples.run_tsp(n, std_saves, algorithm=Algorithm.STANDARD)
    examples.run_tsp(n, ml_saves, algorithm=Algorithm.ML)

# plot GA data
if plot:

    # fitness data
    _, ax_fitness = plt.subplots()
    save_data = SaveData(rnd_saves, std_saves, ml_saves)
    save_data.plot_fitness('Random', ax=ax_fitness)
    save_data.plot_fitness('Standard', ax=ax_fitness)
    save_data.plot_fitness('ML', ax=ax_fitness)

    # classifer performance data
    _, ax_good = plt.subplots()
    _, ax_bad = plt.subplots()
    save_data.plot_performance('ML', ax1=ax_good, ax2=ax_bad)

    # best routes
    _, ax_best = plt.subplots()
    save_data.plot_best('Random', examples.plot_individual_csv, ax_best)

    # show plots
    plt.show()
