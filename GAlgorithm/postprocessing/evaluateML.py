
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILL_COLOR = '#ff975790'
LINE_COLOR = '#e34234'

def plot_classifier_run_data(folder, pattern):
    '''Plots the variables tracked by a ML runs'''
    
    folder = Path(folder)

    df = pd.DataFrame()

    for fp in folder.glob(pattern):
        _df = pd.read_csv(fp, sep='\t', header=1)

        if len(df) == 0:
            df = _df
        else:
            df = pd.concat((df, _df))
    
    # define name of index column for convenience
    index = 'FunctionEvaluation'

    # sort by function evaluation
    df = df.sort_values(index)

    # create 10 bins
    df['quantile'] = pd.qcut(df[index], q=20)

    # group by the bins
    grouped = df.groupby('quantile')

    # find the mean of each bin
    means = grouped.mean()

    # find the standard deviation of each bin
    stds = grouped.std()

    # convenience for accessing columns
    good_perc = 'GoodPredictorPercentage'
    bad_perc = 'BadPredictorPercentage'

    # Upper/lower ounds of good predictor percentages
    good_upper = means[good_perc] + stds[good_perc]
    good_lower = means[good_perc] - stds[good_perc]

    # good predictor figure
    fig, ax = plt.subplots()
    plt.ylim(0,1)
    plt.title('Classifier Accuracy of Predicting Good Children')
    plt.ylabel('Percentage')
    plt.fill_between(means[index], good_upper, good_lower, color=FILL_COLOR)
    means.plot(x=index, y='GoodPredictorPercentage', c=LINE_COLOR, ax=ax)

    # Upper/lower ounds of bad predictor percentages
    bad_upper = means[bad_perc] + stds[bad_perc]
    bad_lower = means[bad_perc] - stds[bad_perc]

    # bad predictor figure
    fig, ax = plt.subplots()
    plt.ylim(0,1)
    plt.title('Classifier Accuracy of Predicting Bad Children')
    plt.ylabel('Percentage')
    plt.fill_between(means[index], bad_upper, bad_lower, color=FILL_COLOR)
    means.plot(x=index, y='BadPredictorPercentage', c=LINE_COLOR, ax=ax)

    # show plots
    plt.show()
    
