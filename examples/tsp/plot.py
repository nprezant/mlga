
import sys
from operator import itemgetter

import matplotlib.pyplot as plt


def plot_histories(pop_histories:list):
    plt.style.use('seaborn-whitegrid')

    # plot all on one axis
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Travelling Salesman Problem')
    ax.set_title('Travelling Salesman Problem')
    plot_all(ax, pop_histories)
    plt.show()


class PlotPoint:
    def __init__(self):
        self.parent = None # parent history this point belongs to
        self.eval = None
        self.gen = None
        self.mean_dist = None
        self.perc90_dist = None
        self.perc10_dist = None

    
    def distance_to(self, other):
        '''Returns the distance from one plot point to another plot point'''
        return ((self.eval - other.eval)**2 + (self.mean_dist - other.mean_dist)**2)**(1/2)



def plot_all(ax, pop_histories):
    '''plots all histories on the same axes'''
    points: PlotPoint = []
    
    for hist,title in pop_histories:
        gens = []
        dist = []
        evals = []
        perc90 = []
        perc10 = []

        for i,pop in enumerate(hist):
            pt = PlotPoint() # define a point that contains all the necessary plotting data
            pt.parent = pop
            pt.gen = i
            pt.mean_dist = pop.mean_fitness
            pt.perc90_dist =  pop.get_percentile(0.90).distance - pt.mean_dist
            pt.perc10_dist = -pop.get_percentile(0.10).distance + pt.mean_dist
            if i==0:
                pt.eval = 0
            else:
                pt.eval = points[-1].eval + pop.f_evals
            points.append(pt)


            gens.append(i)
            dist.append(pop.mean_fitness)
            if i==0: 
                evals.append(0)
            else: 
                evals.append(evals[i-1] + pop.f_evals)

            perc90.append(pop.get_percentile(0.90).distance-dist[i])
            perc10.append(-pop.get_percentile(0.10).distance+dist[i])

        dist_err = ax.errorbar(evals[5::5], dist[5::5], yerr=[perc10[5::5], perc90[5::5]], fmt='.', capsize=2)
        color = dist_err.lines[0].get_color() # Line2D of the line
        _, = ax.plot(evals, dist, ':', color=color, label=title)
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Total Distance')
        #ax.set_xlim(0, max(evals))
    mark, = ax.plot(points[0].eval, points[0].mean_dist, '*', markersize=16, label='Selected Population')
    ax.legend()


    def histories_clicked(event):
        # when axes is clicked, find the nearest line point, and open a figure for the plot_population with that population
        if not event.inaxes == ax: return
        print(f'\nx = {event.xdata}')
        print(f'y = {event.ydata}')

        # define clicked point
        clicked_pt = PlotPoint()
        clicked_pt.eval = event.xdata
        clicked_pt.mean_dist = event.ydata

        # set the closest point equal to the first point
        closest_pt = points[0]
        closest_pt_dist = closest_pt.distance_to(clicked_pt)

        for pt in points[1:]:
            dist = pt.distance_to(clicked_pt)
            if dist < closest_pt_dist:
                closest_pt = pt
                closest_pt_dist = dist

        print(f'predict closest point: eval={closest_pt.eval}, dist={closest_pt.mean_dist}, gen={closest_pt.gen}')
        mark.set_xdata(closest_pt.eval)
        mark.set_ydata(closest_pt.mean_dist)
        plt.draw()
        plot_population(closest_pt.parent)

    ax.figure.canvas.mpl_connect('button_press_event', histories_clicked)


def plot_population(pop):
    '''Plots the routes in a single population'''
    rows = 3
    cols = 3
    fig, axarr = plt.subplots(rows, cols)
    fig.canvas.set_window_title('Route Visualization for Selected Population')

    ranked_pop = pop.ranked
    count = 0
    
    for i in range(rows):
        for j in range(rows):
            ax = axarr[i, j]
            ax.set_title(f'Ranked Route No. {count}')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            route = ranked_pop[count]
            count += 1
            _, = ax.plot(route.x, route.y, 'b*', markersize=12)
            _, = ax.plot(route.x, route.y, 'r--')
            ax.axis('equal')
    plt.show()