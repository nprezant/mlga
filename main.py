
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import examples

from GAlgorithm import (
    SaveLocation,
    SaveData,
    Algorithm
)

n = 3
folder = Path().cwd() / Path(f'data{n}')

rnd_saves = SaveLocation(folder, 'Random')
std_saves = SaveLocation(folder, 'Standard')
ml1_saves = SaveLocation(folder, 'ML_KNN')
ml2_saves = SaveLocation(folder, 'ML_DT')

run = False
plot = True

# generate GA run data
if run:
    examples.run_tsp(n, rnd_saves, algorithm=Algorithm.RANDOM)
    examples.run_tsp(n, std_saves, algorithm=Algorithm.STANDARD)
    examples.run_tsp(n, ml1_saves, algorithm=Algorithm.ML, classifier_class=KNeighborsClassifier)
    examples.run_tsp(n, ml2_saves, algorithm=Algorithm.ML, classifier_class=DecisionTreeClassifier)

# plot GA data
if plot:

    # set up the save data
    save_data = SaveData(rnd_saves, std_saves, ml1_saves, ml2_saves)

    # fitness data
    _, ax_fitness = plt.subplots()
    save_data.plot_fitness('Random', ax=ax_fitness)
    save_data.plot_fitness('Standard', ax=ax_fitness)
    save_data.plot_fitness('ML_KNN', ax=ax_fitness)
    save_data.plot_fitness('ML_DT', ax=ax_fitness)

    # classifer performance data
    _, ax_good = plt.subplots()
    _, ax_bad = plt.subplots()
    save_data.plot_performance('ML_KNN', ax1=ax_good, ax2=ax_bad, quantiles=10)
    save_data.plot_performance('ML_DT', ax1=ax_good, ax2=ax_bad, quantiles=10)

    # best routes
    _, ax_best = plt.subplots()
    save_data.plot_best('Random', examples.plot_individual_csv, ax_best)
    save_data.plot_best('Standard', examples.plot_individual_csv, ax_best)
    save_data.plot_best('ML_KNN', examples.plot_individual_csv, ax_best)
    save_data.plot_best('ML_DT', examples.plot_individual_csv, ax_best)

    # plot stats
    _, ax_stats = plt.subplots()
    save_data.plot_convergence(
        ['Random', 'Standard', 'ML_KNN', 'ML_DT'],
        target=630, ax=ax_stats
    )

    # plot classifier performance summary
    _, ax_perf = plt.subplots()
    save_data.plot_performance_summary(
        ['ML_KNN', 'ML_DT'], ax=ax_perf
    )

    # show plots
    plt.show()
