
import examples

from pathlib import Path
from GAlgorithm import plot_classifier_run_data, plot_fitness_data

n = 100
folder_name = f'dataDT/data{n}'
data_path = Path().cwd() / folder_name
file_patterns = ['RandomRun*', 'StandardRun*', 'MLRun*']

# generate GA run data
# examples.run_tsp_random(n, folder_name)
# examples.run_tsp_standard(n, folder_name)
# examples.run_tsp_ml_mod(n, folder_name)

# plot GA data
plot_fitness_data(data_path, file_patterns)
plot_classifier_run_data(data_path, 'MLClassifierVarsRun*')
