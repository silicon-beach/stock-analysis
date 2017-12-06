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


AMPLITUDE_SIGMA = 0.05
TIME_SIGMA = 0.06
DEFAULT_PATTERN_LENGTH = 70  # Must be multiple of 7

def generate_synthetic_data(patterns):
    """
    Description:
        Generate synthetic data to validate the pattern matching algorithms.
    """
    noisy_data = {}

    for pat_name, pat_data in patterns.items():
        noisy_data[pat_name] = generate_one_pattern(pat_data['x'],pat_data['y'],
                                                    DEFAULT_PATTERN_LENGTH )

    return noisy_data


def generate_one_pattern(pat_data_x, pat_data_y,amp_sigma,time_sigma,pattern_length):
    """
    Description: Generate synthetic data for one pattern
    """
    tmp_x = np.copy(pat_data_x)
    tmp_y = np.copy(pat_data_y)

    tmp_x = time_warping(tmp_x,time_sigma)
    tmp_x,tmp_y = time_scaling(tmp_x,tmp_y,pattern_length)
    tmp_y = noise_adding(tmp_y,amp_sigma)

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



def time_warping(pattern_x,time_sigma):
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
        sigma = time_sigma

        if i>0 and i<numPts-1:
            output_x[i] = random.gauss(pattern_x[i],sigma)
        else:
            #output_x[i] = prevPt + abs(random.gauss(0,sigma))
            output_x[i] = pattern_x[i]



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


def test_pattern_results(pattern_dict,pattern_name,amp_sigma=AMPLITUDE_SIGMA,
                         time_sigma=TIME_SIGMA,pattern_length=DEFAULT_PATTERN_LENGTH):
    """
    Input:
        pattern_dict: Dictionary of patterns.
        pattern_name: Name of pattern to test.

    Description:
        Run test and plot on one pattern
    """
    curr_pat = pattern_dict[pattern_name]
    data_x,data_y = generate_one_pattern(curr_pat['x'],curr_pat['y'],
                                         amp_sigma,time_sigma,
                                         pattern_length)
    pip_y,pip_x = rc.PIP_identification(data_y,data_x)

    plt.plot(data_x,data_y,'-o')
    plt.plot(pip_x,pip_y,'--x')
    plt.title(pattern_name)
    plt.show()


    distortion_val, min_pattern_name = rc.multiple_template_matching(pip_y, pip_x, pattern_dict)
    print('Detected as ' + min_pattern_name)
    print('Distortion value: ' + str(distortion_val))
    if min_pattern_name == pattern_name:
        print('Correct Detection!')
        return True
    else:
        print('Wrong Detection')
        return False

    pattern_len = len(pattern_dict[min_pattern_name]['y'])
    plt.plot(pip_x[:pattern_len],pip_y[:pattern_len],'--x')
    plt.title('PIP Plot: ' + pattern_name)
    plt.show()




if __name__ == '__main__':
    accuracy_count = 0

    template_pat = tp.template_patterns()
    for pat_name in template_pat:
        isCorrect = test_pattern_results(template_pat,pat_name)
        if isCorrect is True:
            accuracy_count += 1


    print('Accuracy: ' + str(accuracy_count) + '/' + str(len(template_pat)))
