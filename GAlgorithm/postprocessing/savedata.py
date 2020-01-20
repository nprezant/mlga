''' Importing data into the dataframe '''

from pathlib import Path

import pandas as pd
from matplotlib.table import Table

from .base import SaveLocation
from .fitness import plot_fitness_df
from .evaluateML import plot_performance_df


class SaveData:
    ''' Provides a one stop shop to access all save data
    Data is "lazy-loaded" -- only loaded when necessary
    '''

    def __init__(self, *args: SaveLocation):
        ''' Each arg is a SaveLocation '''

        self._locations: SaveLocation = {}

        for location in args:
            self.add_location(location)

    def add_location(self, location: SaveLocation):
        ''' Adds this location to the internal dictionary.
        Key is the location base name.
        Converts location to LocationData first.
        '''
        self._locations[location.base_name] = location

    def plot_fitness(self, loc: str, ax=None):
        ''' Plots the fitness of the save data in a given location '''

        # try to get the location of the save data
        location = self._locations[loc]

        # get fitness save data
        df = location.fitness_df

        # plot fitness data
        plot_fitness_df(df, label=location.base_name, ax=ax)

    def plot_performance(self, loc: str, ax1=None, ax2=None):
        ''' Plots the performance of the save data in a given location '''

        # try to get the location of the save data
        location = self._locations[loc]

        # get the performance save data
        df = location.performance_df

        # plot performance data
        plot_performance_df(df, label=location.base_name, ax1=ax1, ax2=ax2)

    def plot_best(self, loc: str, plot_fn, ax=None):
        ''' Plot the best individual of the save data in a given location'''

        # try to get the location of the save data
        location = self._locations[loc]

        # get the df of the best individual
        fp = location.best_individual_fp

        # plot individual data
        plot_fn(fp, label=fp.stem, ax=ax)

    def plot_convergence(self, locs: str, target, ax=None):
        ''' Plot the convergence of the save data in a given location '''

        index = []
        optimums = []
        pop_contains = []

        for loc in locs:
            # try to get the location of the save data
            location: SaveLocation = self._locations[loc]
            location.read_convergence_stats(target)

            index.append(location.base_name)
            optimums.append(location.f_evals_to_converge)
            pop_contains.append(location.f_evals_to_get_target_in_pop)

        # construct dataframe
        df = pd.DataFrame(
            {
                'Converged': optimums,
                'Discovered Optimum': pop_contains,
            },
            index = index
        )

        # plot
        ax = df.plot.bar(rot=0, ax=ax)
        ax.set_xlabel('Algorithm Run')
        ax.set_ylabel('Function Evaluation')
        ax.set_title('Function Evaluations to Converge')

        # write convergence data to csv
        df.to_csv(location.base_folder / Path('_convergence.txt'))
