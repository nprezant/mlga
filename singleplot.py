
import sys
from operator import itemgetter

import matplotlib.pyplot as plt

import savejson


def plotmultipleroutesets(runs):
    # add a figure for each run
    figs =[]
    for run in runs:
        fig = plt.figure()
        fig.canvas.set_window_title('Genetic Algorithm with ML: ' + run['classifier'])
        figs.append(fig)
        routes = run['best_routes']
        plotsinglerouteset(fig, routes)
    plt.show()


def plotroutes(pop_histories:list):
    plt.style.use('seaborn-whitegrid')

    # plot all on one axis
    fig, ax = plt.subplots()
    ax.set_title('Travelling Salesman Problem')
    plotall(ax, pop_histories)

    # plot each individual figure
    for pop_history in pop_histories:
        fig = plt.figure()
        plotsinglerouteset(fig, *pop_history)
    plt.show()


def plotall(ax, pop_histories):
    '''plots all histories on the same axes'''
    
    for hist,title in pop_histories:
        gens = []
        dist = []
        evals = []
        perc90 = []
        perc10 = []

        for i,pop in enumerate(hist):
            gens.append(i)
            dist.append(1/pop.mean_fitness)
            if i==0: 
                evals.append(0)
            else: 
                evals.append(evals[i-1] + pop.f_evals)

            perc90.append(pop.get_percentile(0.90)-dist[i])
            perc10.append(-pop.get_percentile(0.10)+dist[i])

        dist_err = ax.errorbar(evals[5::5], dist[5::5], yerr=[perc10[5::5], perc90[5::5]], fmt='.', capsize=2)
        dist_line, = ax.plot(evals, dist, ':', label=title)
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Total Distance')
        ax.set_xlim(0, max(evals))
    ax.figure.legend()



def plotsinglerouteset(fig, pop_history, title):
    '''plots the stuff'''
    
    ax1 = fig.add_axes([0.55, 0.53, 0.35, 0.35]) # route plot
    ax2 = fig.add_axes([0.15, 0.10, 0.75, 0.35]) # generation plot
    ax3 = fig.add_axes([0.15, 0.53, 0.35, 0.35]) # text box

    # plot data
    gens = []
    dist = []
    evals = []
    perc90 = []
    perc10 = []

    for i,pop in enumerate(pop_history):
        gens.append(i)
        dist.append(1/pop.mean_fitness)
        if i==0: 
            evals.append(0)
        else: 
            evals.append(evals[i-1] + pop.f_evals)

        perc90.append(pop.get_percentile(0.90)-dist[i])
        perc10.append(-pop.get_percentile(0.10)+dist[i])

    # index of route with minimum distance
    imin = min(enumerate(dist), key=itemgetter(1))[0]

    # plot text box
    base_text = f'Optimal Generation: {imin}\n\n'

    title = plt.text(0.10, 0.90, base_text,
                    bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax3.transAxes, ha='left', va='top')

    # city data plot
    route = pop_history[imin].best_individual
    cityMarks, = ax1.plot(route.x, route.y, 'b*', markersize=12)
    routeLine, = ax1.plot(route.x, route.y, 'r--')
    ax1.axis('equal')

    # route distance lines
    dist_err = ax2.errorbar(evals[5::5], dist[5::5], yerr=[perc10[5::5], perc90[5::5]], fmt='.', capsize=2)
    dist_line, = ax2.plot(evals, dist, ':')
    dist_mark, = ax2.plot(evals[imin], dist[imin], '*', markersize=16, zorder=20)
    ax2.set_xlabel('Function Evaluations')
    ax2.set_ylabel('Total Distance')
    ax2.set_xlim(0, max(evals))
    

    def update(eval):
        '''updates the plot
        *gen* is the generation to set the stuff to'''

        # slider value comes off as float
        eval = int(eval)
        gen = feval2gen(eval, evals, gens)
        title.set_text(base_text + f'Function Evaluation: {eval}\n'
                                   f'Generation: {gen}')

        route = pop_history[gen].mean_individual
        if route is None: return

        # plot of route through cities
        routeLine.set_xdata(route.x)
        routeLine.set_ydata(route.y)

        # plot of distance over generation
        dist_mark.set_xdata(evals[gen])
        dist_mark.set_ydata(route.distance)

        # update canvas
        fig.canvas.draw_idle()
        return None


    def ax2_on_click(event):
        if not event.inaxes == ax2: return
        update(event.xdata)

    fig.canvas.mpl_connect('button_press_event', ax2_on_click)

    return fig


def pt_in_axes(ax, x, y):
    '''Returns true is the x,y point is inside the axes.
    x, y must be in figure coordinates'''
    is_inside = False
    (left, bottom, right, top) = axes2figbox_lbrt(ax, ax.figure)
    if left < x < right:
        if bottom < y < top:
            is_inside = True
    return is_inside

    
def axes2figbox_lbrt(ax, fig):
    '''Returns the axis location in figure coordinates
    [left, bottom, right, top]'''

    # npbox is [ [left, bottom], [right, top] ]
    npbox = fig.transFigure.inverted().transform(ax.patch.get_extents())

    left = npbox[0][0]
    bottom = npbox[0][1]
    right = npbox[1][0]
    top = npbox[1][1]

    return [left, bottom, right, top]


def axes2figbox_lbwh(ax, fig):
    '''Returns the axis location in figure coordinates
    [left, bottom, width, height]'''

    (left, bottom, right, top) = axes2figbox_lbrt(ax, fig)

    width = right-left
    height = top-bottom

    return [left, bottom, width, height]


def feval2gen(f_eval:int, evals:list, gens:list):
    '''Converts number of function evaluations to the generation
    If the number can't be found, it returns the first generation'''
    for i,num_evals in enumerate(evals):
        if num_evals >= f_eval:
            return gens[i]
    return gens[0]


if __name__ == '__main__':
    dir = sys.argv[1]
    pattern = sys.argv[2]
    runs = savejson.load_from_files(dir, pattern)
    plotmultipleroutesets(runs)
    #fp = sys.argv[1]
    #dct = savejson.load(fp)
    #plotroutes(dct['best_routes'])