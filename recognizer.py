#!/bin/python3
import numpy as np


def PIP_identification(P, Q_length=7):
    """
    Input:
            P: input sequence
            Q_length: length of query sequence; Defaults : 7
    Output:
            returns a sequence of Perceptually important point (PIP) identification of size Q_length
    """
    try:
        SP = [-1] * Q_length
        SP[0] = P[0]
        SP[Q_length - 1] = P[-1]
        counter = 1
        index_mid = int((Q_length - 1) / 2)
        SP[index_mid], index_lower_end = maximize_PIP_distance(P)
        index_upper_start = index_lower_end
        index_lower_start = 0
        index_upper_end = len(P) - 1

        while counter < index_mid:
            SP[index_mid - counter], index_lower_end = maximize_PIP_distance(
                P[index_lower_start:index_lower_end + 1])

            SP[index_mid + counter], index_temp = maximize_PIP_distance(
                P[index_upper_start:index_upper_end + 1])

            index_upper_start += index_temp
            counter += 1

        return SP
    except ValueError:
        return []


def maximize_PIP_distance(P):
    """
    Input:
            P: input sequence
    Output:
            returns a point with maximum distance to P[1] and P[-1] 
    """
    P1 = [1, P[0]]
    P2 = [len(P), P[-1]]

    distance_P1_P2 = ((P2[1] - P1[0]) ** 2 + (P2[0] - P1[0]) ** 2) ** 0.5

    np_perpendicular_dist = np.fromiter((perpendicular_distance(
        P1, P2, [xi + 1, P[xi]], distance_P1_P2) for xi in range(1, len(P) - 1)), np.float64)

    index_max = 1 + np.argmax(np_perpendicular_dist)

    return P[index_max], index_max


def perpendicular_distance(P1, P2, P3, distance):
    """
    Input:
            P1,P2,P3: 3 points in [x,y] format
            S: slope between P1 and P2; this can pre-calculated
    Output:
            returns (perpendicular distance) between these 3 points
    """

    PD = abs((P2[1] - P1[1]) * P3[0] - (P2[0] - P1[0]) *
             P3[1] + P2[0] * P1[1] - P2[1] * P1[0]) / distance

    return PD
