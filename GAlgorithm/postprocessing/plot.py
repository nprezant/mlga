
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLORS = [
    '#ff881f',
    '#36a436',
    '#3c89bd',
]

cindex = 0

def plot_fitness_df(df, label='Fitness', ax=None):

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
        ax.fill_between(x=means[index], y1=means['90th Percentile'], y2=means['10th Percentile'], color=color, alpha=0.3)
        means.plot(x=index, y='Mean Fitness', c=color, ax=ax_arg, label=label)

        # axis settings
        ax.set_title('Fitness Curves')
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Fitness')
        