# The Benefits of Machine Learning Classification within the Genetic Algorithm Evolutionary Cycle:

*A Case Study of the Traveling Salesman Problem*

# Introduction

This project serves to demonstrate the effect that machine learning classifiers can have on the efficiency of genetic algorithms. The standard genetic algorithm process is shown in Figure 1. The modified algorithm process is shown in Figure 2. Note how the standard algorithm passes *all* children to the evaluation procedure, while the modified algorithm only passes *some* children to be evaluated (only the children deemed “good” by the machine learning classifier).

<img src="/media/image1.png" width="500" />

Figure 1, Standard Genetic Algorithm

<img src="/media/image2.png" width="500" />

Figure 2, Modified Genetic Algorithm with Machine Learning Classification

The efficiency of an algorithm is often indicated by the total number of **objective function** calls. That is, the total number of times that an algorithm must evaluate an individual’s fitness before the optimal solution is reached. For long and complex objective functions each individual evaluation could be on the order of minutes, making it desirable to minimize the number of objective function calls.

One clear inefficiency of the genetic algorithm is that the evolutionary process generates many bad designs, with only a few improving from the parent population. Evaluating these designs takes a large amount of computational time, but in the standard genetic algorithm the only way to *know* that the design is bad is to first evaluate it.

The machine learning classifier attempts to solve this problem by effectively pre-sorting the children and discarding children that it expects to have poor fitness. Therefore, the only children that get evaluated are those that pass through the classifier. The classifier is very fast – a fraction of a second to train, and a fraction of a second to predict new classifications. If the training data grows very large, the time it takes to train the classifier will certainly increase, but this should remain negligible when compared to the long running objective function calls.

# Setup

The traveling salesmen problem is a combinational optimization problem that answers the following question: “given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city and returns to the origin city?”[^1] The traveling salesmen problem is one of the most studied problems in optimization and is used in this paper as a base for genetic algorithm exploration; i.e., various genetic algorithm modifications will be applied to the same traveling salesmen problem to determine/compare the performance of each modification.

## City Layout

Figure 3 shows the layout of the cities that salesmen must travel through, with the optimum route traced though it. Ten cities were randomly generated on a 200x200 grid, and the fitness is computed as the length of the route in grid units (though any number of cities on any size grid could be used; the values here were chosen to decrease overall computation time). The city layout was consistent for all genetic algorithm run data in this report. As an example, Figure 4 shows a non-optimal route through the cities.

<img src="/media/image3.png" width="300" />

Figure 3, Traveling Salesmen City Layout with Optimal Route Traced

<img src="/media/image4.png" width="300" />

Figure 4, Traveling Salesmen City Layout with Non-Optimal Route



## Parameter List

The following tables describe the algorithm’s input parameters and the values used in this report. Table 1 lists the inputs to the standard genetic algorithm. Table 2 lists the additional parameters necessary for the modified genetic algorithm to consume.

Table 1, Standard Genetic Algorithm Parameters

| Parameter                      | Value used in this experiment                                |
| ------------------------------ | ------------------------------------------------------------ |
| Initial Population             | \[Random set of 100 routes through default cities\]          |
| Fitness Function               | \[Finds total distance of route\]                            |
| Tournament Selection Size      | 2                                                            |
| Mutation Rate (%)              | 5%                                                           |
| Mutation Function              | \[Randomly swaps order of two cities with each other\]       |
| Crossover Function             | \[Takes segment from one parent, randomly distributes other cities around that segment\] |
| Max \# of Function Evaluations | 50,000                                                       |

Table 2, Additional Machine Learning Genetic Algorithm Parameters

<table><thead><tr class="header"><th>Parameter</th><th>Value used in this experiment</th></tr></thead><tbody><tr class="odd"><td>Training Data Function</td><td>[Converts salesmen route into a numerical list of city coordinates. E.g. [67, 56, 197, 20, …]]</td></tr><tr class="even"><td>Classifier Threshold*</td><td>25%</td></tr><tr class="odd"><td>Classifier Method</td><td>[One of the following: K Nearest Neighbors, Decision Tree, Gaussian Naïve Bayes]</td></tr></tbody></table>
<table><tbody><tr class="even"><td><p><span class="underline">Table Notes:</span></p><p>* The classifier is passed training data in the form of:</p><table><thead><tr class="header"><th>“good”</th><th>[route coordinates]</th></tr></thead><tbody><tr class="odd"><td>“bad”</td><td>[route coordinates]</td></tr><tr class="even"><td>…</td><td></td></tr></tbody></table><p>Each time the modified genetic algorithm completes a generation, the population (already evaluated for fitness) is saved in a “population history.” To train the classifier, each individual in the population history is sorted by fitness and categorized based on the input parameter classifier threshold. That is, if the classifier percentage is 25%, and there are 100 individuals in the population history, then the best 25 will be classified as “good” and the rest will be classified as “bad”.</p></td><td></td></tr></tbody></table>
# Algorithm Performance

Figure 5 shows a comparison of the standard algorithm with the modified algorithm. A “random” algorithm is also shown for reference. The random algorithm finds solutions by randomly generating 100 individuals at a time, evaluating them, then culling with the previous 100 to maintain population size. Each algorithm was run separately a total of 100 times each and Figure 5 displays the overall mean fitness values and distributions of those runs.

<img src="/media/image5.png" width="300" />

Figure 5, Fitness Curve Comparison of Genetic Algorithms

The shaded region surrounding each fitness curve represent the fitness of the 90th percentile and 10th percentile individuals. (E.g. to find the 90th percentile individual out of a population of 100, rank the individuals by fitness, and take the individual at index 90.)

Both genetic algorithms are clearly superior to the random algorithm, and the algorithm using the machine learning classification clearly outperforms the standard genetic algorithm. Interestingly, the spread of the machine learning algorithm is consistently much narrower than the spread of the standard genetic algorithm. The details of *how much* better one algorithm is than another are shown in Figure 6.

**Convergence** is defined as the number of function evaluations it took for the mean population to be within 1% of the optimum.

The **Discovered Optimum** is the number of function evaluations taken for the optimum to be within 1% of the 10th/90th bounds.

The machine learning algorithm both discovers the optimum solution sooner (by ≈6,000 function evaluations) and, once it has found the optimum, converges on it sooner (by about ≈1,000 function evaluations).

<img src="/media/image6.png" width="300" />

Figure 6, Convergence Comparison of Genetic Algorithms

# Classifier Performance

Machine learning algorithms can have shortcomings. For the genetic algorithm application, there are two obvious mistakes that can be made: 1) the classifier can **incorrectly discard good solutions**, or 2) the classifier can **incorrectly let bad solutions through**. While these mistakes sound similar, they have different implications.

When the classifier incorrectly discards good solutions, it is actively preventing the genetic algorithm from benefiting from those solutions. At the best, it would simply decrease the algorithm’s overall efficiency by giving the algorithm more children to evaluate. At the worst, this could cause premature convergence.

Letting bad solutions through is less of an issue, for it only gives the algorithm more children to evaluate.

These inaccuracies were tracked and can be seen plotted in Figure 7. For this experiment, we defined classifier performance by determining how well the classifier predicted good children and how well the classifier predicted bad children. Example calculations are shown below.

<span class="underline">Predicting good children:</span>

> Population size: 100  
> Classifier categorizes 30 children as “good”  
> 10 of those children are actually “bad” (their fitness is below the 25% fitness threshold)  
> Therefore, the classifier’s accuracy at predicting good children is 20/30 ≈ 67%

<span class="underline">Predicting bad children:</span>

> Population size: 100  
> Classifier categorizes 70 children as “bad”  
> 5 of those children are actually “good” (their fitness is above the 25% fitness threshold)  
> Therefore, the classifier’s accuracy at predicting bad children is 65/70 ≈ 93%

## Classifier Performance: Observed

The comparison of average classifier performance between the K-Nearest Neighbors, Gaussian Naïve Bayes, and Decision Tree algorithms is plotted in Figure 7. Detailed tracking of the classifier prediction performance over the course of the population’s evolution is shown in Figure 8.

<img src="/media/image7.png" width="400" />

Figure 7, Average Classifier Performance Comparison

While the Gaussian Naïve Bayes algorithm (shown in Figure 7) clearly has the greatest accuracy in predicting good children, it also has the worst accuracy predicting bad children. As discussed previously, the ability of the classifier to accurately predict *bad* children is more critical to the classifier’s success, as failing to do so results in the algorithm discarding useful child solutions. Though, the Naïve Bayes algorithm is *much* better at predicting good children than the others, while only *slightly* worse at predicting bad children. Different problem types will have different optimum classifiers, and the choice of classifier should be left to the user as an algorithm input parameter.

<img src="/media/image8.png" width="300" /><img src="/media/image9.png" width="300" />

Figure 8, Detailed Classifier Performance Comparison

Figure 8 shows the mean classifier performance over the course of an algorithm run, with the shaded region indicating the ± one standard deviation of the classifier’s performance around that function evaluation (recall that this data represents a mean of 100 separate algorithm runs). The machine learning classifiers, while resulting in different performance values, appear to have very similar trends over the course of the run. The classifier’s ability to accurately classify good children starts off very poor – this makes sense, as the training data starts out very small (the first classifier is trained with just the initial population). Though, from there, the classifier quickly reaches its asymptotic limit (both Decision Tree and KNN reach their limit before 2000 function evaluations, while Naïve Bayes takes until around 4000 function evaluations, albeit leveling off at almost twice the accuracy of the others).

All classifiers start out able to very accurately classify bad children: \>95%. All classifiers also take a quick 5-10% drop in accuracy shortly after the start of the run. From there, both the Decision Tree and KNN classifiers improve as the run progresses (as expected). But interestingly the Naïve Bayes classifier *gets worse* at determining bad children over the course of run. This means that further into the run, the classifier is more likely to discard a good solution than it is at the beginning.[^2]

Despite these differences in accuracy, there is little discrepancy in the actual fitness of the results (Figure 9). If we were to find the area under these fitness curves, we would likely find that the Naïve Bayes classifier slightly outperformed the KNN and Decision Tree classifiers. All classifiers both discover the optimum solution and converge upon it with roughly the same number of function evaluations (Figure 10). Again, Naïve Bayes slightly outperforms the others.

<img src="/media/image10.png" width="400"/>

Figure 9, Fitness Curve Comparison Across Machine Learning Classifiers

<img src="/media/image11.png" width="400" />

Figure 10, Convergence Comparison Across Machine Learning Classifiers

Although KNN and Decision Tree were both more accurate at predicting bad children, Naïve Bayes ultimately found and converged upon the optimal solution sooner, making it “the best” classifier in this experiment. However, this is specific to the optimization problem and won’t necessarily remain the best classifier when applied to different problems (or even the same problem if the training data was organized differently).

I am also personally disappointed in this result, as I found the Naïve Bayes algorithm runs to take significantly longer on my machine.[^3]

# Conclusions

The genetic algorithm with machine learning classification is significantly more efficient than a genetic algorithm without. For example, if one function evaluation took 30 seconds of computational time, and (as shown in Figure 10) the modified algorithm took 6,000 fewer evaluations to converge, that would save 50 hours’ worth of computational time.

Changing the machine learning classifier does little to influence the convergence rates of the algorithm, even when the classifiers have largely differing classification accuracies. There may be some benefit achieved by tuning the classifier to best match the problem, but no significant differences came of that in this study.

[^1]: Wikipedia: <https://en.wikipedia.org/wiki/Travelling_salesman_problem>

[^2]: The classifier performance is currently tracked on a binary, low fidelity scale, such that the classifier would be marked completely wrong for classifying an individual that was *just barely* not good enough. When manually stepping through the code, I saw that misclassifications were often *just barely* in nature.

For example:
    
* A given set of training data has a 40% cutoff fitness threshold of 880 km.
  
* One of the children generated has a fitness of 879 km.
  
* The classifier classified this child as “bad” and was penalized for it when calculating its accuracy (since technically, the child was better than the threshold, and “good”).

Future experiments could include some kind of “good faith tolerance” when calculating classifier performance such that these small misclassifications are not weighted the same as a more egregious error. Perhaps misclassifications within 5% of the threshold could be tracked as a separate variable, or even better, the percent of how far the misclassified individual was from the threshold could be tracked as a continuous variable and plotted separately. This way, we could see the distribution of where most “Classified as Bad but Actually Good” (and vice versa) individuals fall on the fitness scale relative to their threshold value.

[^3]: Genetic algorithms will sometimes converge pre-maturely on local optimum, and not be able to find a path from that optimum to the global optimum if they are too far apart in the solution space. The algorithm in this experiment had a feature that would give an “extra kick” to the population whenever it appeared to converge (determined by how much progress had been made in the most recent generations). This extra kick effectively shakes the population up by greatly increasing the mutation rate in an attempt to dislodge itself from a local optimum.

I noticed that the Naïve Bayes classifier tended to invoke this feature *much* more often than the other classifiers which may have led to longer computational time on my machine while maintaining low objective function counts overall. 

Future experiments could track the number of times this feature is invoked in each population. From that data, one could see whether any one classifier is more likely to converge locally than another, and also whether this local convergence combined with the “extra kick” is correlated with lower function evaluations overall.