# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:40:31 2017

@author: Tan_4
"""

import numpy as np
import random
import template_patterns as tp
import matplotlib.pyplot as plt
import recognizer as rc


NOISE_SIGMA = 0.03
PATTERN_LENGTH = 70

def generate_synthetic_data(patterns):
    """
    Description:
        Generate synthetic data to validate the pattern matching algorithms.
    """
    noisy_data = {}

    for pat_name, pat_data in patterns.items():
        noisy_data[pat_name] = generate_one_pattern(pat_data['x'],pat_data['y'])

    return noisy_data


def generate_one_pattern(pat_data_x, pat_data_y):
    """
    Description: Generate synthetic data for one pattern
    """
    tmp_x = np.copy(pat_data_x)
    tmp_y = np.copy(pat_data_y)

    tmp_x = time_warping(tmp_x)
    tmp_x,tmp_y = time_scaling(tmp_x,tmp_y,PATTERN_LENGTH)
    tmp_y = noise_adding(tmp_y,NOISE_SIGMA)

    return tmp_x,tmp_y


def time_scaling(pattern_x, pattern_y, numOfPts):
    """
    Input:
        pattern_x: The pattern to scale, x-axis.
        pattern_y: The pattern to scale, y-axis.
        numOfPts: Number of points to scale to.
    Output:
        Tuple containing the new X points, new Y points
    Description:
        Scales the pattern by increasing the number of data points in between
        current existing points.
    """

    min_pt = pattern_x[0]
    max_pt = pattern_x[-1]

    new_x_points = np.linspace(min_pt,max_pt,numOfPts)
    new_y_points = np.interp(new_x_points,pattern_x,pattern_y)

    return (new_x_points, new_y_points)



def time_warping(pattern_x):
    """
    Input:
        pattern_x: The pattern to apply transformation, x-axis.
    Output:
        The transformed pattern, x-axis.
    Description:
        Apply time warping, which will shift the time axis for the pattern data.
    """

    numPts = len(pattern_x)
    output_x = np.zeros(numPts)

    for i in range(numPts):
        currPt = pattern_x[i]
        prevPt = pattern_x[i-1] if i>0      else currPt
        nextPt = pattern_x[i+1] if i<numPts-1 else currPt
        sigma = (nextPt - prevPt) / 4

        if i>0 and i<numPts-1:
            output_x[i] = random.gauss(currPt,sigma)
        else:
            #output_x[i] = prevPt + abs(random.gauss(0,sigma))
            output_x[i] = currPt



    return output_x


def noise_adding(pattern_y, noise_sigma):
    """
    Input:
        pattern_y: The pattern to apply transformation, y-axis.
        noise_variance: Variance value input to the random normal distribution.
    Output:
        The transformed pattern, y-axis
    Description:
        Apply noise to the y data, using random sample from normal distribution
        based on the given variance.
    """

    numPts = len(pattern_y)
    output_y = np.zeros(numPts)

    for i in range(numPts):
        currPt = pattern_y[i]
        output_y[i] = random.gauss(currPt,noise_sigma)

    return output_y


if __name__ == '__main__':

    template_pat = tp.template_patterns()
    noisy_data = generate_synthetic_data(template_pat)

    accuracy_count = 0

    for pat_name, pat_data in noisy_data.items():
        pip_y,pip_x = rc.PIP_identification(pat_data[1],pat_data[0])

        plt.plot(pat_data[0],pat_data[1],'-o')
        plt.plot(pip_x,pip_y,'--x')

        plt.title(pat_name)
        plt.show()

        plt.plot(pip_x,pip_y,'--x')
        plt.title('PIP Plot: ' + pat_name)
        plt.show()

        distortion = []

        for template_name, template_data in template_pat.items():
            val = rc.template_matching(pip_y,pip_x,
                                       template_data['y'],template_data['x'])
            #print('Distortion (' + template_name + '): ' + str(val))

            distortion.append((val,template_name))

        minIdx = np.argmin(distortion,axis=0)[0]
        if minIdx > len(distortion):
            print("Less than 0")
        print('Detected as ' + str(distortion[minIdx]))
        if (distortion[minIdx])[1] == pat_name:
            print('Correct Detection!')
            accuracy_count += 1
        else:
            print('Wrong Detection')


    print('Accuracy: ' + str(accuracy_count) + '/' + str(len(noisy_data)))

