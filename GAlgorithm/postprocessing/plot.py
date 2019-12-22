
from pathlib import Path
from collections import UserList

from ..plot import PlotPoints, fitness_plot_from_points


def plot_many_objective_files(folder, run_file_patterns, mean=True):

    folder = Path(folder)

    if mean == True:
        plot_objective_files_mean(folder, run_file_patterns)
    else:
        for pattern in run_file_patterns:
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


def objective_files_mean_points(folder, pattern):

    mean_points = PlotPoints()

    for fp in folder.glob(pattern):

        # read the file
        points = PlotPoints()
        points.read_csv(fp)

        # interpolate at each evaluation
        points.interp(1)

        # add these points to the mean
        mean_points.add_to_mean(points)

    return mean_points


def plot_objective_file_mean(folder, pattern):

    mean_points = objective_files_mean_points(folder, pattern)

    fitness_plot_from_points([(mean_points, 'Means')], pattern)


def plot_objective_files_mean(folder, patterns):

    points_list = []

    for pattern in patterns:
        mean_points = objective_files_mean_points(folder, pattern)
        points_list.append((mean_points, f'Mean of {pattern}'))

    fitness_plot_from_points(points_list, 'GA Comparison')


# TODO 2) import and plot ML classifier variables
