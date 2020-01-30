# A Study of the Benefits of Machine Learning Classification in the Genetic Algorithm Evolutionary Cycle

This package explores the benefits of machine learning classification in the genetic algorithm's evolutionary loop.

## Overview



## Traveling Salesman Example

While the package can handle multiple problem types, we focus on the traveling salesmen, a minimization problem that seeks to find the shortest route through a set of points.

### Overall Effectiveness

At first glance, the genetic algorithm with the machine learning classifier shows significant improvement over the standard genetic algorithm.

![Travelling Salesman](assets/travelling_salesman.png)

### Route Comparison

The top 10 salesman routes can be compared with and without machine learning (at 1000 objective function evaluations). The machine learning routes are notably consistent and better (shorter).
![Routes Comparison](assets/routes_comparison.png)

### Classifier Effectiveness

Genetic algorithms often generate many children that are worse solutions than the parent.

The classifier's job is to determine which of the children are good or better than the parents, and which are worse. The genetic algorithm can then simply discard the "bad" children and not waste computation time evaluating their fitness.

The machine learning classifier in this experiment must be evaluated on it's effectiveness and accuracy. If it classifies many actual good children as bad, then the genetic algorithm will not see those children and suffer for it. If it classifies many actual bad children as good, that is more acceptable as it will not worsen the actual solution results, but it will mean that we are losing efficiency and spending needless time on objective function evaluations.
