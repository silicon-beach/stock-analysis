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


NOISE_X_MEAN = 20
NOISE_X_SIGMA = 15
NOISE_Y_MEAN = 0.5
NOISE_Y_SIGMA = 0.1




def generate_series_of_synthetic_data(patterns, series_tuple):
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

    for pat_name,pat_length in series_tuple:
        pat_data = patterns[pat_name]
        tmp_x, tmp_y = gsd.generate_one_pattern(pat_data['x'],pat_data['y'])

        series_y.extend(tmp_y.tolist())


        # Random noise in between patterns

        series_y.extend(generate_random_noise())

    series_x = np.linspace(0,1,len(series_y))
    series_y = np.array(series_y)
    series_y = series_y / series_y.max() # Normalise between 0 to 1

    return series_x,series_y

def generate_random_noise():

    numPts = int(round(random.gauss(NOISE_X_MEAN,NOISE_X_SIGMA)))
    noise_values = np.random.normal(NOISE_Y_MEAN,NOISE_Y_SIGMA,numPts)
    return noise_values.tolist()



if __name__ == '__main__':
    patterns = tp.template_patterns()
    series_tuple = []
    series_tuple.append(('head and shoulders',70))
    series_tuple.append(('inverse head and shoulders',98))

    series_x,series_y = generate_series_of_synthetic_data(patterns, series_tuple)

    plt.plot(series_x,series_y,'-o')









