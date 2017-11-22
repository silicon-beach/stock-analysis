# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:50:15 2017

@author: Damien
"""

import random
import math

from deap import base
from deap import creator
from deap import tools

import recognizer as rc
import template_patterns as tp
import matplotlib.pyplot as plt

MAX_GENERATIONS = 50
MIN_FITNESS = 0.1

# Desired segment length
DLEN = 70
POPULATION_SIZE = 50
CROSSOVER_RATE = 0.6
SELECTION_RATE = 1
SEL_TOURNAMENT_SIZE = 5


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


    pop = toolbox.population(n=POPULATION_SIZE)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(MAX_GENERATIONS):
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
        # Mutate by adding element
        isRepeated = True
        randNum = -1
        while isRepeated is True:
            randNum = random.randrange(1,maxVal)
            if randNum not in chromo:
                isRepeated = False

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

    distortion_sum = 0
    ind_len = len(ind)
    for i in range(ind_len+1):
        startIdx = ind[i-1] if i>0 else 0
        endIdx   = ind[i] if i<ind_len else len(data_x)

        tmp_x = data_x[startIdx:endIdx]
        tmp_y = data_y[startIdx:endIdx]

        if(len(tmp_y) == 0):
            print("This should not happen" + str(startIdx) + ' ' + str(endIdx))

        pip_y,pip_x = rc.PIP_identification(tmp_y,tmp_x)

        distortion_val, pattern_name = rc.multiple_template_matching(
                                       pip_y, pip_x, TEMPLATE_PATTERNS)
        distortion_sum += distortion_val


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

            print('Detected as ' + pattern_name)
            print('Distortion value: ' + str(distortion_val))
            #input('Press any key to continue...')

    # Normalize the distortion value by num of segments
    distortion_sum /= (ind_len+1)

    return (distortion_sum,)


if __name__ == '__main__':
    pass



