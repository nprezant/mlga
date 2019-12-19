
from pathlib import Path
from collections import UserList

import numpy as np

from ..plot import PlotPoints, fitness_plot_from_points


# TODO 1) import and plot run data
def plot_many_objective_files(folder, run_file_patterns, mean=True):

    folder = Path(folder)

    for pattern in run_file_patterns:
        if mean == True:
            plot_objective_files_mean(folder, pattern)
        else:
            plot_objective_files(folder, pattern)


def plot_objective_files(folder, pattern):

    points_list = []

    for fp in folder.glob(pattern):

        points = PlotPoints()
        points.read_csv(fp)

        points_list.append((points, fp.stem))

    fitness_plot_from_points(points_list, pattern)


def plot_objective_files_interpolations(folder, pattern):

    points_list = []

    for fp in folder.glob(pattern):

        # read the file
        points = PlotPoints()
        points.read_csv(fp)

        # interpolate at each evaluation
        points.interp(1)

        points_list.append((points, fp.stem))

    fitness_plot_from_points(points_list, pattern)

def plot_objective_files_mean(folder, pattern):

    mean_points = PlotPoints()

    for fp in folder.glob(pattern):

        # read the file
        points = PlotPoints()
        points.read_csv(fp)

        # interpolate at each evaluation
        points.interp(1)

        # add these points to the mean
        mean_points.add_to_mean(points)

    fitness_plot_from_points([(mean_points, 'Means')], pattern)


# TODO 2) import and plot ML classifier variables