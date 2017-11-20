# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:49:45 2017

@author: Tan_4
"""

import matplotlib.pyplot as plt
import numpy as np
import json

def template_patterns():
    """
    Description:
        The template patterns are defined here.
    """

    TEMPLATE_SIZE = 7

    patterns = {}

    patterns['head and shoulders'] = {}
    patterns['head and shoulders']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['head and shoulders']['y'] = np.array([0,2,1.5,2.5,1.5,2,0])

    patterns['inverse head and shoulders'] = {}
    patterns['inverse head and shoulders']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['inverse head and shoulders']['y'] = (-patterns['head and shoulders']['y'] +
                                                  np.max(patterns['head and shoulders']['y']))

    patterns['double top'] = {}
    patterns['double top']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['double top']['y'] = np.array([0,1,2,1.1,2,1,0])

    patterns['inverse double top'] = {}
    patterns['inverse double top']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['inverse double top']['y'] = (-patterns['double top']['y'] +
                                           np.max(patterns['double top']['y']))

    patterns['triple top'] = {}
    patterns['triple top']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['triple top']['y'] = np.array([0,2,1.1,2,1.1,2,0])

    patterns['inverse triple top'] = {}
    patterns['inverse triple top']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['inverse triple top']['y'] = (-patterns['triple top']['y'] +
                                           np.max(patterns['triple top']['y']))

    patterns['rounded top'] = {}
    patterns['rounded top']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['rounded top']['y'] = np.array([0,1,1.08,1.12,1.08,1,0])

    patterns['inverse rounded top'] = {}
    patterns['inverse rounded top']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['inverse rounded top']['y'] = (-patterns['rounded top']['y'] +
                                           np.max(patterns['rounded top']['y']))

    patterns['spike top'] = {}
    patterns['spike top']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['spike top']['y'] = np.array([0,0.1,0,1,0,0.1,0])

    patterns['inverse spike top'] = {}
    patterns['inverse spike top']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['inverse spike top']['y'] = (-patterns['spike top']['y'] +
                                           np.max(patterns['spike top']['y']))


    # Normalise the patterns to be between 0 and 1 for amplitude
    for pat_name, pat_data in patterns.items():
        pat_data['y'] /= np.max(pat_data['y'])

    return patterns



# Plot the patterns if running this file.
if __name__ == '__main__':

    patterns = template_patterns()

    for pat_name, pat_data in patterns.items():
        plt.plot(pat_data['x'],pat_data['y'])
        plt.title(pat_name)
        plt.show()

    for pat_name in patterns:
        patterns[pat_name]['x'] = patterns[pat_name]['x'].tolist()
        patterns[pat_name]['y'] = patterns[pat_name]['y'].tolist()


    json_content = json.dumps(patterns)
    f = open('patterns.json', 'w')
    f.write(json_content)
    f.close()




