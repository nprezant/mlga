
import matplotlib.pyplot as plt

class PlotPoint:
    
    def __init__(self):
        self.parent = None # parent history this point belongs to
        self.eval = None
        self.gen = None
        self.mean = None
        self.perc90 = None
        self.perc10 = None

    def distance_to(self, other):
        '''Returns the distance from one plot point to another plot point'''
        return ((self.eval - other.eval)**2 + (self.mean - other.mean)**2)**(1/2)


def fitness_plot(population_history:list, title='Fitness Plot'):
    '''Plots the fitness vs function evaluation count.
    Plots for each population in the `population_history` list'''
    plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots()
    fig.canvas.set_window_title(title)
    ax.set_title(title)

    for hist,label in population_history:

        points: PlotPoint = []
        
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
                pt.eval = points[-1].eval + pop.f_evals
            points.append(pt)

        evals = [pt.eval for pt in points]
        vals = [pt.value for pt in points]
        perc10s = [pt.perc10 for pt in points]
        perc90s = [pt.perc90 for pt in points]

        # dist_err = ax.errorbar(evals[5::5], vals[5::5], yerr=[perc10s[5::5], perc90s[5::5]], fmt='.', capsize=2)
        # color = dist_err.lines[0].get_color() # Line2D of the line
        _, = ax.plot(evals, vals, ':', label=label)
        ax.fill_between(evals, [v-p for v,p in zip(vals,perc10s)], [v+p for v,p in zip(vals,perc90s)], alpha=0.5)
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Population Fitness')
        #ax.set_xlim(0, max(evals))
    #mark, = ax.plot(points[0].eval, points[0].value, '*', markersize=16, label='Selected Population')
    ax.legend()

    plt.show()