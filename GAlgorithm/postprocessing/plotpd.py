
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLORS = [
    '#ff881f',
    '#36a436',
    '#3c89bd',
]


def plot_fitness_data(folder, patterns: list):
    '''Plots the variables tracked by a ML runs'''
    
    folder = Path(folder)

    _, ax = plt.subplots()
    color_index = 0

    # for each input pattern
    for pattern in patterns:

        df = pd.DataFrame()

        # for each file that matches the pattern
        for fp in folder.glob(pattern):
            _df = pd.read_csv(fp, sep='\t', header=0)

            if len(df) == 0:
                df = _df
            else:
                df = pd.concat((df, _df))

        # define name of index column for convenience
        index = 'Function Evaluations'

        # sort by function evaluation
        df = df.sort_values(index)

        # create bins
        df['quantile'] = pd.qcut(df[index], q=30)

        # group by the bins
        grouped = df.groupby('quantile')

        # find the mean of each bin
        means = grouped.mean()

        # get next color
        try:
            color = COLORS[color_index]
        except IndexError:
            color_index = 0
            color = COLORS[color_index]
        finally:
            color_index += 1

        # plot
        plt.fill_between(x=means[index], y1=means['90th Percentile'], y2=means['10th Percentile'], color=color, alpha=0.3)
        means.plot(x=index, y='Mean Fitness', c=color, ax=ax, label=pattern)

    plt.show()