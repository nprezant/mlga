
from pathlib import Path

import matplotlib.pyplot as plt

import examples

from GAlgorithm import (
    plot_classifier_run_data,
    plot_fitness_data,
    SaveLocation,
    Algorithm
)

n = 1
folder = Path().cwd() / Path(f'data{n}')

rnd_saves = SaveLocation(folder, 'RandomRun', 'BestOfRandomRun', 'PerformanceOfRandomRun')
std_saves = SaveLocation(folder, 'StandardRun', 'BestOfStandardRun', 'PerformanceOfStandardRun')
ml_saves = SaveLocation(folder, 'MLRun', 'BestOfMLRun', 'PerformanceOfMLRun')

run = False
plot = True

# generate GA run data
if run:
    examples.run_tsp(n, rnd_saves, algorithm=Algorithm.RANDOM)
    examples.run_tsp(n, std_saves, algorithm=Algorithm.STANDARD)
    examples.run_tsp(n, ml_saves, algorithm=Algorithm.ML)

# plot GA data
# if plot:
#     _, ax = plt.subplots()
#     plot_fitness_data(data_path, ['StandardRun*', 'RandomRun*'])
#     plot_fitness_data(data_path, file_patterns)
#     plot_classifier_run_data(data_path, 'MLClassifierVarsRun*')
