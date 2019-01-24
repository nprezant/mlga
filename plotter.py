
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.widgets import Slider
from matplotlib.text import Text


def plotroutes2(routes):
    '''plots a list of routes as an animation'''
    fig, (ax1, ax2) = plt.subplots(2, 1)

    title = ax1.text(0.5,0.5, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
            transform=ax1.transAxes, ha="center")

    cityMarks, = ax1.plot(routes[0].x, routes[0].y, 'b*')
    routeLine, = ax1.plot(routes[0].x, routes[0].y, 'r--')

    gen = [i for i in range(len(routes))]
    dist = [r.distance for r in routes]
    frontier, = ax2.plot(gen, dist, '-')
    frontierTrace, = ax2.plot(gen[0], dist[0], '*', markersize=12)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Total Distance')


    def update(frame, routeLine, routes, frontierTrace):
        '''updates the plot'''

        title.set_text(f'Generation: {frame}\n'
                       'Route distance: {:8.2f}\n'.format(routes[frame].distance) +
                       'Initial Distance: {:8.2f}'.format(routes[0].distance))

        routeLine.set_xdata(routes[frame].x)
        routeLine.set_ydata(routes[frame].y)

        frontierTrace.set_xdata(frame)
        frontierTrace.set_ydata(routes[frame].distance)

        return routeLine, title, frontierTrace


    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=len(routes), fargs=(routeLine, routes, frontierTrace), interval=200, blit=True)

    plt.show()


def plotroutes(routes):
    '''plots a list of routes as an with user controls'''
    fig, (ax1, ax2) = plt.subplots(2, 1)

    title = plt.text(0.5, 1.1, 'Generation 0',
                    bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax1.transAxes, ha='center')

    cityMarks, = ax1.plot(routes[0].x, routes[0].y, 'b*', markersize=12)
    routeLine, = ax1.plot(routes[0].x, routes[0].y, 'r--')

    gens = [i for i in range(len(routes))]
    dist = [r.distance for r in routes]
    frontier, = ax2.plot(gens, dist, '-')
    frontierTrace, = ax2.plot(gens[0], dist[0], '*', markersize=16)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Total Distance')
    ax2.set_xlim(0, len(gens))
    

    def update(gen):
        '''updates the plot
        *gen* is the generation to set the stuff to'''
        gen = int(gen)
        title.set_text(f'Generation: {gen}\n'
                       'Route distance: {:8.2f}\n'.format(routes[gen].distance) +
                       'Initial Distance: {:8.2f}'.format(routes[0].distance))

        routeLine.set_xdata(routes[gen].x)
        routeLine.set_ydata(routes[gen].y)

        frontierTrace.set_xdata(gen)
        frontierTrace.set_ydata(routes[gen].distance)

        fig.canvas.draw_idle()
        return None
    
    axcolor = 'lightgoldenrodyellow'

    box = axes2figbox(ax2, fig)

    axgen = plt.axes(box, facecolor=axcolor, alpha=0.5, visible=False)

    sgen = Slider(axgen, '', valmin=0, valmax=len(gens)-1, valinit=0) #valstep =1)
    sgen.valtext.set_visible(False)
    sgen.on_changed(update)

    plt.show()

    
def axes2figbox(ax, fig):
    '''Returns the axis location in figure coordinates
    [left, bottom, width, height]'''

    # npbox is [ [left, bottom], [right, top] ]
    npbox = fig.transFigure.inverted().transform(ax.patch.get_extents())

    left = npbox[0][0]
    bottom = npbox[0][1]
    right = npbox[1][0]
    top = npbox[1][1]

    width = right-left
    height = top-bottom

    return [left, bottom, width, height]