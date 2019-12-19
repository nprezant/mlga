
from collections import UserList

import numpy as np
import matplotlib.pyplot as plt

class PlotPoint:
    
    def __init__(self):
        self.parent = None # parent history this point belongs to
        self.eval = None
        self.gen = None
        self.value = None
        self.perc90 = None
        self.perc10 = None

    def distance_to(self, other):
        '''Returns the distance from one plot point to another plot point'''
        return ((self.eval - other.eval)**2 + (self.value - other.value)**2)**(1/2)


class PlotPoints(UserList):
    
    @property
    def evals(self):
        return [p.eval for p in self.data]

    @evals.setter
    def evals(self, evals):
        for p, eval in zip(self.data, evals):
            p.eval = eval

    @property
    def values(self):
        return [p.value for p in self.data]

    @values.setter
    def values(self, values):
        for p, value in zip(self.data, values):
            p.value = value

    @property
    def gens(self):
        return [p.gen for p in self.data]

    @gens.setter
    def gens(self, gens):
        for p, gen in zip(self.data, gens):
            p.gen = gen

    @property
    def perc10s(self):
        return [p.perc10 for p in self.data]

    @perc10s.setter
    def perc10s(self, perc10s):
        for p, perc10 in zip(self.data, perc10s):
            p.perc10 = perc10

    @property
    def perc90s(self):
        return [p.perc90 for p in self.data]

    @perc90s.setter
    def perc90s(self, perc90s):
        for p, perc90 in zip(self.data, perc90s):
            p.perc90 = perc90

    def create_from_ga_history(self, hist):
        self.data:PlotPoint = []

        for i,pop in enumerate(hist):
            pt = PlotPoint() # define a point that contains all the necessary plotting data
            pt.parent = pop
            pt.gen = i
            pt.value = pop.mean_fitness
            pt.perc90 =  pop.get_percentile(0.90).fitness - pt.value
            pt.perc10 = -pop.get_percentile(0.10).fitness + pt.value
            if i==0:
                pt.eval = 0
            else:
                pt.eval = self.data[-1].eval + pop.f_evals
            self.data.append(pt)

    def read_csv(self, fp):
        self.data:PlotPoint = []

        with open(fp, 'r') as f:
            f.readline()
            for line in f:
                pt = PlotPoint()
                vals = line.split('\t')

                pt.eval = int(vals[0])
                pt.gen = int(vals[1])
                pt.value = float(vals[2])
                pt.perc90 = float(vals[3])
                pt.perc10 = float(vals[4])
                self.data.append(pt)

    def csv_headers(self) -> str:
        return (
            'Function Evaluations\tGeneration\tMean Fitness\t'
            '90th Percentile\t10th Percentile\r'
        )

    def write_csv(self, fp, mode='w'):
        with open(fp, mode) as f:
            for pt in self.data:
                f.write(
                    f'{pt.eval}\t{pt.gen}\t{pt.value}\t'
                    f'{pt.perc90}\t{pt.perc10}\r'
                )


    def interp(self, step):
        '''Interpolates *in place* properties at each evaluation step'''

        # create x values to interpolate at
        x = np.arange(0, self.evals[-1], step)

        # interpolate
        interp_values = np.interp(x, self.evals, self.values)
        interp_gens = np.interp(x, self.evals, self.gens)
        interp_perc10s = np.interp(x, self.evals, self.perc10s)
        interp_perc90s = np.interp(x, self.evals, self.perc90s)

        # create new data list
        points_data = []
        for i, eval in enumerate(x):
            p = PlotPoint()
            p.eval = eval
            p.value = interp_values[i]
            p.gen = interp_gens[i]
            p.perc10 = interp_perc10s[i]
            p.perc90 = interp_perc90s[i]
            points_data.append(p)

        # save results in place
        self.data = points_data


    def add_to_mean(self, points):
        '''Adds a new set of points to this one, averaging them together'''

        def averageLists(list1, list2):

            # find the mean of the items that overlap
            mean_of_overlap = [(a+b)/2 for a,b in zip(list1, list2)]

            # find the extra bit that did not overlap
            extra_length = abs(len(list1) - len(list2))

            if extra_length == 0:
                extra_segment = []
            elif len(list1) > len(list2):
                extra_segment = list1[-extra_length:]
            else:
                extra_segment = list2[-extra_length:]

            # combine overlapping segment and extra segment
            return mean_of_overlap + extra_segment

        if len(self.data) == 0:
            self.data = points.data
        else:
            self.evals = averageLists(self.evals, points.evals)
            self.gens = averageLists(self.gens, points.gens)
            self.values = averageLists(self.values, points.values)
            self.perc10s = averageLists(self.perc10s, points.perc10s)
            self.perc90s = averageLists(self.perc90s, points.perc90s)


def fitness_plot(population_history:list, title='Fitness Plot'):
    '''Plots the fitness vs function evaluation count.
    Plots for each population in the `population_history` list
    '''

    points_list = []
    for hist,label in population_history:

        points = PlotPoints()
        points.create_from_ga_history(hist)
        points_list.append((points, label))

    fitness_plot_from_points(points_list, title)

def fitness_plot_from_points(points_list, title):
    '''points_list given as [(Points(), 'Plot 1'), (Points(), 'Plot 2')]'''

    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(title)
    ax.set_title(title)

    for points, label in points_list:
        evals = [pt.eval for pt in points]
        vals = [pt.value for pt in points]
        perc10s = [pt.perc10 for pt in points]
        perc90s = [pt.perc90 for pt in points]

        _, = ax.plot(evals, vals, ':', label=label)
        ax.fill_between(evals, [v-p for v,p in zip(vals,perc10s)], [v+p for v,p in zip(vals,perc90s)], alpha=0.5)
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Population Fitness')

    ax.legend()

    plt.show()