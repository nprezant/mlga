
import sys
from operator import itemgetter

import matplotlib.pyplot as plt
#import matplotlib.animation as animation

#from matplotlib.widgets import Slider, TextBox
#from matplotlib.text import Text

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


def plotroutes(routes):
    fig = plt.figure()
    plotsinglerouteset(fig, routes)
    plt.show()



def plotsinglerouteset(fig, routes):
    '''plots a list of routes as an with user controls'''
    
    ax1 = fig.add_axes([0.55, 0.53, 0.35, 0.35]) # route plot
    ax2 = fig.add_axes([0.15, 0.10, 0.75, 0.35]) # generation plot
    ax3 = fig.add_axes([0.15, 0.53, 0.35, 0.35]) # text box

    # frontier plot data
    gens = [i for i in range(len(routes))]
    dist = [r.distance for r in routes]

    # index of route with minimum distance
    imin = min(enumerate(dist), key=itemgetter(1))[0]

    # plot text box
    base_text = f'Optimal Generation: {imin}\n\n'

    title = plt.text(0.10, 0.90, base_text,
                    bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax3.transAxes, ha='left', va='top')

    # city data plot
    cityMarks, = ax1.plot(routes[imin].x, routes[imin].y, 'b*', markersize=12)
    routeLine, = ax1.plot(routes[imin].x, routes[imin].y, 'r--')
    ax1.axis('equal')

    # frontier lines
    frontier, = ax2.plot(gens, dist, '-')
    frontierTrace, = ax2.plot(gens[imin], dist[imin], '*', markersize=16)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Total Distance')
    ax2.set_xlim(0, len(gens))
    

    def update(gen):
        '''updates the plot
        *gen* is the generation to set the stuff to'''

        # slider value comes off as float
        gen = int(gen)
        title.set_text(base_text + ''
                       f'Generation: {gen}\n'
                       'Route distance: {:8.2f}\n'.format(routes[gen].distance) +
                       'Initial Distance: {:8.2f}'.format(routes[0].distance))

        # plot of route through cities
        routeLine.set_xdata(routes[gen].x)
        routeLine.set_ydata(routes[gen].y)

        # plot of distance over generation
        frontierTrace.set_xdata(gen)
        frontierTrace.set_ydata(routes[gen].distance)

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


if __name__ == '__main__':
    dir = sys.argv[1]
    pattern = sys.argv[2]
    runs = savejson.load_from_files(dir, pattern)
    plotmultipleroutesets(runs)
    #fp = sys.argv[1]
    #dct = savejson.load(fp)
    #plotroutes(dct['best_routes'])