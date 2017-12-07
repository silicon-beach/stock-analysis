# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:50:15 2017

@author: Damien
"""

import random
import math
import numpy as np

from deap import base
from deap import creator
from deap import tools

import recognizer as rc
import template_patterns as tp
import matplotlib.pyplot as plt

POPULATION_SIZE = 50
MAX_GENERATIONS = 50

# Desired segment length
DLEN = 70
CROSSOVER_RATE = 0.5
SELECTION_RATE = 1
SEL_TOURNAMENT_SIZE = 10
MIN_SEGMENT_LENGTH = 10


# Probability to add a datapoint during mutate
# If mutate does not add a point, it will drop a point.
MUTATE_ADD_PROB = 0.5
MUTATE_PROB = 1 - CROSSOVER_RATE

# Stores all template patterns to be matched.
TEMPLATE_PATTERNS = tp.template_patterns()


def runGA(data_x,data_y):
    """
    Input:
        data_x: Data values, x-axis(time)
        data_y: Data values, y-axis
    Description: Runs the GA algorithm on the data
    """

    # I assume that the data is a 2D nested list, with first dim for each row of data.
    numDataPts = len(data_x)
    segmentCount = int(math.ceil(numDataPts / DLEN)) - 1

    # Convert to numpy format
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", set, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Each chromosome has size: segmentCount, with values between 0 to numDataPts
    # Generate permutations with no repetition
    toolbox.register("indices", random.sample, range(1,numDataPts-1), segmentCount)
    toolbox.register("individual",tools.initIterate, creator.Individual,
                 toolbox.indices)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate",tools.cxMessyOnePoint)
    toolbox.register("mutate",my_mutate)
    toolbox.register("evaluate",evaluate,data_x,data_y)

    toolbox.register("select", tools.selTournament)

    # For gathering statistics
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.nanmean)
    stats_fit.register("std", np.nanstd)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.nanmax)

    stats_size = tools.Statistics(key=len)
    stats_size.register("avg", np.mean)
    stats_size.register("std", np.std)
    stats_size.register("min", np.min)
    stats_size.register("max", np.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    logbook = tools.Logbook()

    pop = toolbox.population(n=POPULATION_SIZE)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(MAX_GENERATIONS):
        # Statistics collection
        record = mstats.compile(pop)
        print("Generation: " + str(g))
        #print(record['fitness'])
        logbook.record(gen=g,**record)

        # Select the next generation individuals
        offspring = toolbox.select(pop, int(round(len(pop) * SELECTION_RATE)),
                                   SEL_TOURNAMENT_SIZE)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))


        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:

                # Crossover requires list, so convert out of set data strcture
                childlist1 = list(child1)
                childlist2 = list(child2)
                toolbox.mate(childlist1, childlist2)
                child1.clear()
                child2.clear()
                child1.update(childlist1)
                child2.update(childlist2)
                del child1.fitness.values
                del child2.fitness.values


        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < MUTATE_PROB:
                toolbox.mutate(mutant,numDataPts)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    print(logbook)
    with open('GA_log.txt','w') as f:
        f.write(logbook.__str__())

    plot_statistics(logbook)



    # Return the best individual
    return (tools.selBest(pop,1))[0]

def my_mutate(chromo,maxVal):
    """
    Input:
            chromo: Chromosome
            maxVal: Largest value +1 the chromosome can take
    Output:
            returns a sequence of Perceptually important point (PIP)
            identification and the corresponding time values of size Q_length
    """

    if (random.random() < MUTATE_ADD_PROB and len(chromo) < maxVal):
        chromo_list = list(chromo)
        # Mutate by adding element

        disallowed_values = set()
        for val in chromo_list:
            if val-MIN_SEGMENT_LENGTH+1 < 0:
                lower_bound = 0
            else:
                lower_bound = val-MIN_SEGMENT_LENGTH+1

            if val+MIN_SEGMENT_LENGTH > maxVal:
                upper_bound = maxVal
            else:
                upper_bound = val+MIN_SEGMENT_LENGTH

            disallowed_values.union(range(lower_bound,upper_bound))

        allowed_values = set(range(1,maxVal)) - disallowed_values
        randNum = random.sample(allowed_values,1)[0]
        chromo.add(randNum)

    elif len(chromo) > 0:
        # Mutate by removing element
        valueToDel = random.sample(chromo,1)[0]
        chromo.remove(valueToDel)


def evaluate(data_x,data_y,ind_set,plot_data=False):
    """
    Description:
        Calculates the fitness value by using the template method.
    """

    # Get a list from the set data structure
    ind = sorted(ind_set)


    # Find number of different template lengths that exist
    pat_len = set()
    for pat in TEMPLATE_PATTERNS:
        pat_len.add(len(pat))

    pat_len = sorted(pat_len)


    distortion_sum = 0
    ind_len = len(ind)
    for i in range(ind_len+1):
        startIdx = ind[i-1] if i>0 else 0
        endIdx   = ind[i] if i<ind_len else len(data_x)

        tmp_x = data_x[startIdx:endIdx]
        tmp_y = data_y[startIdx:endIdx]

        if(len(tmp_y) == 0):
            print("This should not happen" + str(startIdx) + ' ' + str(endIdx))

        pip_y,pip_x = rc.PIP_identification(tmp_y,tmp_x,7,isNumpyFormat=True)

        distortion_val, pattern_name = rc.multiple_template_matching(
                                       pip_y, pip_x, TEMPLATE_PATTERNS)

        # Treat values above threshold as pattern not detected
        #if distortion_val > 0.2:
        #    distortion_val = 5

        #distortion_val += rc.temporal_control_penalty(endIdx-startIdx,70,2)

        distortion_sum += distortion_val

        # Early exit for invalid chromosomes.
        if np.isinf(distortion_sum) == True:
            break


        # Plot for debugging
        if plot_data == True:
            plt.plot(data_x,data_y)
            plt.axvline(x=data_x[startIdx],linestyle='--')
            plt.axvline(x=data_x[endIdx-1],linestyle='--')
            plt.plot(pip_x,pip_y,'-x')
            plt.title('Data Plot: ' + str(startIdx) + ' - ' + str(endIdx))
            plt.show()

            plt.plot(pip_x,pip_y,'-x',color='c')
            plt.title('PIP Plot: ' + str(startIdx) + ' - ' + str(endIdx))
            plt.show()

            if distortion_val > 0.3:
                print('No Detection.')
            else:
                print('Detected as ' + pattern_name)
            print('Distortion value: ' + str(distortion_val))
            #input('Press any key to continue...')

    # Normalize the distortion value by num of segments
    if np.isinf(distortion_sum) == False:
        distortion_sum /= (ind_len+1)

    return (distortion_sum,)



def amplitude_control_penalty(PIP,dfr,dac):
    """
    Description:

    """
    min_pip = min(PIP)
    fr = (max(PIP) - min_pip) / min_pip
    d2 = fr - dfr
    theta2 = dfr / dac
    AC = 1 - (1/(1+math.exp(-d2/theta2)))

    return AC


def plot_statistics(logbook):
    gen = logbook.select("gen")
    fit_mins = logbook.chapters["fitness"].select("min")
    size_avgs = logbook.chapters["size"].select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()
    plt.savefig('stats_plot.pdf')



if __name__ == '__main__':
    pass



