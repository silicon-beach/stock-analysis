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
    patterns['head and shoulders']['y'] = np.array([0,1.5,1,2,1,1.5,0])

    patterns['inverse head and shoulders'] = {}
    patterns['inverse head and shoulders']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    patterns['inverse head and shoulders']['y'] = (-patterns['head and shoulders']['y'] +
                                                  np.max(patterns['head and shoulders']['y']))
#
#    patterns['double top'] = {}
#    patterns['double top']['x'] = np.linspace(0,1,5)
#    patterns['double top']['y'] = np.array([0,1.5,1,1.5,0])
#
#    patterns['inverse double top'] = {}
#    patterns['inverse double top']['x'] = np.linspace(0,1,5)
#    patterns['inverse double top']['y'] = (-patterns['double top']['y'] +
#                                           np.max(patterns['double top']['y']))
#
#    patterns['triple top'] = {}
#    patterns['triple top']['x'] = np.linspace(0,1,7)
#    patterns['triple top']['y'] = np.array([0,2,1.1,2,1.1,2,0])
#
#    patterns['inverse triple top'] = {}
#    patterns['inverse triple top']['x'] = np.linspace(0,1,7)
#    patterns['inverse triple top']['y'] = (-patterns['triple top']['y'] +
#                                           np.max(patterns['triple top']['y']))
#
#    patterns['rounded top'] = {}
#    patterns['rounded top']['x'] = np.linspace(0,1,7)
#    patterns['rounded top']['y'] = np.array([0,0.9,1,1.12,1,0.9,0])
#
#    patterns['inverse rounded top'] = {}
#    patterns['inverse rounded top']['x'] = np.linspace(0,1,7)
#    patterns['inverse rounded top']['y'] = (-patterns['rounded top']['y'] +
#                                           np.max(patterns['rounded top']['y']))
#
#    patterns['spike top'] = {}
#    patterns['spike top']['x'] = np.linspace(0,1,7)
#    patterns['spike top']['y'] = np.array([0,0.1,0,1,0,0.1,0])
#
#    patterns['inverse spike top'] = {}
#    patterns['inverse spike top']['x'] = np.linspace(0,1,7)
#    patterns['inverse spike top']['y'] = (-patterns['spike top']['y'] +
#                                           np.max(patterns['spike top']['y']))
##
#    patterns['pennant'] = {}
#    patterns['pennant']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['pennant']['y'] = np.array([0,1,0.2,0.8,0.3,0.7,0.4,0.6,1])
#
#    patterns['inverse pennant'] = {}
#    patterns['inverse pennant']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['inverse pennant']['y'] = (-patterns['pennant']['y'] +
#                                           np.max(patterns['pennant']['y']))
#
#    patterns['boardening'] = {}
#    patterns['boardening']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['boardening']['y'] = patterns['inverse pennant']['y'][::-1]
#
#    patterns['inverse boardening'] = {}
#    patterns['inverse boardening']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['inverse boardening']['y'] = (-patterns['boardening']['y'] +
#                                           np.max(patterns['boardening']['y']))
#
#    patterns['diamond'] = {}
#    patterns['diamond']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['diamond']['y'] = np.array([0,0.6,0.4,0.3,0.8,0.3,0.6,0.4,0.9])
#
#    patterns['inverse diamond'] = {}
#    patterns['inverse diamond']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['inverse diamond']['y'] = (-patterns['diamond']['y'] +
#                                           np.max(patterns['diamond']['y']))
#
#    patterns['flag'] = {}
#    patterns['flag']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['flag']['y'] = np.array([0,0.6,0.4,0.6,0.4,0.6,0.4,0.6,1])
#
#    patterns['inverse flag'] = {}
#    patterns['inverse flag']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['inverse flag']['y'] = (-patterns['flag']['y'] +
#                                           np.max(patterns['flag']['y']))
#
#    patterns['wedge'] = {}
#    patterns['wedge']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['wedge']['y'] = np.array([0,0.8,0.4,0.6,0.3,0.4,0.2,0.5,0.8])
#
#    patterns['inverse wedge'] = {}
#    patterns['inverse wedge']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['inverse wedge']['y'] = (-patterns['wedge']['y'] +
#                                           np.max(patterns['wedge']['y']))
#
#    patterns['wedge2'] = {}
#    patterns['wedge2']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['wedge2']['y'] = np.array([0,0.1,0.4,0.1,0.4,0.1,0.4,0.1,0.5])
#
#    patterns['inverse wedge2'] = {}
#    patterns['inverse wedge2']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['inverse wedge2']['y'] = patterns['wedge2']['y'][::-1]
#
#    patterns['uptrend'] = {}
#    patterns['uptrend']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['uptrend']['y'] = np.linspace(0,1,TEMPLATE_SIZE)
#
#    patterns['downtrend'] = {}
#    patterns['downtrend']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
#    patterns['downtrend']['y'] = np.linspace(1,0,TEMPLATE_SIZE)

    #patterns['sideways'] = {}
    #patterns['sideways']['x'] = np.linspace(0,1,TEMPLATE_SIZE)
    #patterns['sideways']['y'] = np.repeat(0.5,TEMPLATE_SIZE)


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




