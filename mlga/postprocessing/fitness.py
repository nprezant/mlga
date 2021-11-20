
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .plotoptions import COLORS

# define name of index column for convenience
INDEX = 'Function Evaluations'

cindex = 0

def _get_fitness_df_means(df, quantiles=30):
    ''' Convert the fitness dataframe to a dataframe of means.
    Split evently between param quantiles.'''

    # sort by function evaluation
    df = df.sort_values(INDEX)

    # create bins
    df['quantile'] = pd.qcut(df[INDEX], q=quantiles)

    # group by the bins
    grouped = df.groupby('quantile')

    # find the mean of each bin
    means = grouped.mean()

    return means

def plot_fitness_df(df, label='Fitness', ax=None, quantiles=30):

    # get means
    means = _get_fitness_df_means(df, quantiles)

    # get color
    global cindex
    try:
        color = COLORS[cindex]
    except IndexError:
        cindex = 0
        color = COLORS[cindex]
    finally:
        cindex += 1

    # set plot commands based on input axis
    if ax is None:
        ax = plt
        ax_arg = None
    else:
        ax_arg = ax

    # plot
    ax.fill_between(x=means[INDEX], y1=means['90th Percentile'], y2=means['10th Percentile'], color=color, alpha=0.3)
    means.plot(x=INDEX, y='Mean Fitness', c=color, ax=ax_arg, label=label)

    # axis settings
    ax.set_title('Fitness Curves')
    ax.set_xlabel('Function Evaluations')
    ax.set_ylabel('Fitness')
        
def convergence_stats(df, target, tolerance=0.10, quantiles=30):
    ''' Gets convergence statistics
    Returns (f evals to converge, f evals to get target in population)
    Target value is optimum fitness
    '''

    # get means
    means = _get_fitness_df_means(df, quantiles)

    def get_first_row_with(condition, df):
        for index, row in df.iterrows():
            if condition(row):
                return index, row
        return None, None # Condition not met on any row in entire DataFrame

    def mean_is_optimum(row):
        fitness = row['Mean Fitness']
        top = fitness + fitness * tolerance
        bottom = fitness - fitness * tolerance
        if target < top and target > bottom:
            return True
        else:
            return False

    def target_in_population(row):
        perc10 = row['10th Percentile']
        perc90 = row['90th Percentile']

        if perc10 < perc90:
            bottom = perc10 - perc10 * tolerance
            top = perc90 + perc90 * tolerance
        else:
            top = perc10 + perc10 * tolerance
            bottom = perc90 - perc90 * tolerance
            
        if target < top and target > bottom:
            return True
        else:
            return False

    # find f_eval num when the mean is within 10% of the target value
    _, row = get_first_row_with(mean_is_optimum, means)
    if row is None:
        f_evals_to_converge = np.nan
    else:
        f_evals_to_converge = row['Function Evaluations']

    # find f_eval num when target value is within population
    _, row = get_first_row_with(target_in_population, means)
    if row is None:
        f_evals_to_get_target_in_pop = np.nan
    else:
        f_evals_to_get_target_in_pop = row['Function Evaluations']

    return f_evals_to_converge, f_evals_to_get_target_in_pop
