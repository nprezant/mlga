''' Standard classes and methods '''

from pathlib import Path
from enum import Enum

import pandas as pd


class Algorithm(Enum):
    STANDARD = 1
    ML = 2
    RANDOM = 3


class SaveLocation:

    def __init__(
        self, 
        base_folder, 
        base_name,
        params_file='_params.txt',
    ):
        self.base_folder = Path(base_folder)
        self.base_name = base_name

        self.run_file = Path(f'{base_name}Run')
        self.best_file = Path(f'BestOf{base_name}Run')
        self.performance_file = Path(f'PerformanceOf{base_name}Run')
        self.params_file = params_file

        self.params_file = Path(params_file)

        self.opened = False

        # data frames
        self.reset_dataframes()

        # best file paths
        self.reset_best_filepaths()
    
    # file paths
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

    # data frames
    def reset_dataframes(self):
        self._fitness_df = None
        self._performance_df = None

    @property
    def fitness_df(self):
        if self._fitness_df is None:
            self._fitness_df = read_files(
                self.base_folder,
                self.run_file.stem + '*',
                header=0
            )
        return self._fitness_df

    @property
    def performance_df(self):
        if self._performance_df is None:
            self._performance_df = read_files(
                self.base_folder,
                self.performance_file.stem + '*',
                header=1
            )
        return self._performance_df

    # best individual file path
    def reset_best_filepaths(self):
        self._best_individual_fp = None

    @property
    def best_individual_fp(self):
        if self._best_individual_fp is None:

            # get the fitness dataframe
            fit_df = self.fitness_df

            # index and file name of the run with the best fitness
            idx = fit_df[['Mean Fitness']].idxmin() # TODO make applicable to max fitness applications too
            f_name = fit_df.at[int(idx), 'FileName']
            
            # get run number from the best fitness file
            base_len = len(self.run_file.stem)
            run_number = f_name[base_len:]

            # make the name of the best individual file
            best_fp = self.best_fp(run_number)
            self._best_individual_fp = best_fp

        return self._best_individual_fp


def read_files(folder, pattern, **kwargs) -> pd.DataFrame:

    folder = Path(folder)

    df = pd.DataFrame()

    for fp in folder.glob(pattern):
        _df = pd.read_csv(fp, sep='\t', **kwargs)
        _df['FileName'] = fp.stem

        if len(df) == 0:
            df = _df
        else:
            df = pd.concat((df, _df))

    return df
    