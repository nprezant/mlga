''' Standard classes and methods '''

from pathlib import Path
from enum import Enum


class Algorithm(Enum):
    STANDARD = 1
    ML = 2
    RANDOM = 3


class SaveLocation:

    def __init__(
        self, 
        base_folder, 
        run_prefix, 
        best_prefix, 
        performance_prefix,
        params_file='_params.txt',
    ):
        self.base_folder = Path(base_folder)
        self.run_file = Path(run_prefix)
        self.best_file = Path(best_prefix)
        self.performance_file = Path(performance_prefix)
        self.params_file = Path(params_file)
    
    @property
    def params_fp(self):
        return self.base_folder / self.params_file

    def run_fp(self, n):
        return self._number(self.run_file, n)

    def best_fp(self, n):
        return self._number(self.best_file, n)

    def performance_fp(self, n):
        return self._number(self.performance_file, n)

    def _number(self, name, n):
        return self.base_folder / f'{name.stem}{n}{name.suffix}'
