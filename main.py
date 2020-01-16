
import examples

# examples.run_tsp_comparison()
# examples.run_tsp_standard(100)
# examples.run_tsp_ml_mod(100)
# examples.run_tsp_random(100)

from pathlib import Path
from GAlgorithm import plot_many_objective_files, plot_classifier_run_data

data_path = Path().cwd() / 'data'
file_patterns = ['StandardRun*', 'MLRun*', 'RandomRun*']

# plot_many_objective_files(data_path, file_patterns)
plot_classifier_run_data(data_path, 'MLClassifierVarsRun*')
