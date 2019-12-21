
import examples

# examples.run_tsp_comparison()
# examples.run_tsp_standard(20)
# examples.run_tsp_ml_mod(20)
# examples.run_tsp_random(20)

from pathlib import Path
from GAlgorithm import plot_many_objective_files

plot_many_objective_files(Path().cwd() / 'data', ['StandardRun*', 'MLRun[0-9]*', 'RandomRun*'])