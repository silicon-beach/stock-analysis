# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:22:39 2017

@author: Tan_4
"""

import generate_synthetic_data as gsd
import template_patterns as tp
import random
import numpy as np
import matplotlib.pyplot as plt
import genetic_algo as ga


NOISE_X_MEAN = 70
NOISE_X_SIGMA = 15
NOISE_Y_MEAN = 0.5
NOISE_Y_SIGMA = 0.01

AMPLITUDE_SIGMA = 0.05
TIME_SIGMA = 0.06


def generate_series_of_synthetic_data(patterns, series_tuple, amp_sigma,
                                      time_sigma,noise_params):
    """
    Input:
        patterns: Dictionary of patterns
        series_tuple: List of tuples, which will store the pattern type and
                      pattern length to generate
    Description:
        Generate a series of patterns. For use in testing with GA version of
        algorithm.
    """

    series_y = []
    #series_y.extend(generate_random_noise(noise_params))

    for pat_name,pat_length in series_tuple:
        pat_data = patterns[pat_name]
        tmp_x, tmp_y = gsd.generate_one_pattern(pat_data['x'],pat_data['y'],
                                                amp_sigma, time_sigma, pat_length)

        series_y.extend(tmp_y.tolist())


        # Random noise in between patterns
        #series_y.extend(generate_random_noise(noise_params))

    series_x = np.linspace(0,1,len(series_y))
    series_y = np.array(series_y)
    series_y = series_y / series_y.max() # Normalise between 0 to 1

    return series_x,series_y

def generate_random_noise(noise_params):

    numPts = abs(int(round(random.gauss(noise_params['x_mean'],noise_params['x_sigma']))))
    noise_values = np.random.normal(noise_params['y_mean'],noise_params['y_sigma'],numPts)
    return noise_values.tolist()




if __name__ == '__main__':
    patterns = tp.template_patterns()
    series_tuple = []
    series_tuple.append(('head and shoulders',70))
    series_tuple.append(('inverse head and shoulders',98))
    series_tuple.append(('head and shoulders',40))
    series_tuple.append(('inverse head and shoulders',200))
    series_tuple.append(('head and shoulders',56))
    series_tuple.append(('inverse head and shoulders',33))
    series_tuple.append(('head and shoulders',100))
    series_tuple.append(('inverse head and shoulders',40))

    noise_params = {}
    noise_params['x_mean'] = NOISE_X_MEAN
    noise_params['x_sigma'] = NOISE_X_SIGMA
    noise_params['y_mean'] = NOISE_Y_MEAN
    noise_params['y_sigma'] = NOISE_Y_SIGMA

    series_x,series_y = generate_series_of_synthetic_data(patterns,
                                                          series_tuple,
                                                          AMPLITUDE_SIGMA,
                                                          TIME_SIGMA,
                                                          noise_params)

    plt.plot(series_x,series_y,'-o')
    plt.show()

    segments = ga.runGA(series_x,series_y)

    ga.evaluate(series_x,series_y,segments,True)








