
import examples

# examples.run_tsp_comparison()
# examples.run_tsp_standard(10)
# examples.run_tsp_ml_mod(3)

from pathlib import Path
from GAlgorithm import plot_many_objective_files

plot_many_objective_files(Path().cwd() / 'data', ['StandardRun*', 'MLRun?.txt'])