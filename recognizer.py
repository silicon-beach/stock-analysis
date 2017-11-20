#!/bin/python3
import numpy as np

TEMPLATE_OMEGA_WT = 0.5


def PIP_identification(P, P_time, Q_length=7):
    is_pip = [False] * len(P)
    is_pip[0] = True
    is_pip[-1] = True
    perp_distance = [-1] * len(P)
    SP = []
    SP_time = []
    for i in xrange(1, Q_length - 1):
        perp_distance, index = PIP_distance(is_pip, P, perp_distance)
        is_pip[index] = True

    for i in xrange(0, len(P)):
        if is_pip[i]:
            SP.append(P[i])
            SP_time.append(P_time[i])

    return SP, SP_time


def get_adjacent_pip_index(index, is_pip, side):
    k = index
    if side == "right":
        while not is_pip[k]:
            k += 1
    if side == "left":
        while not is_pip[k]:
            k -= 1
    return k


def PIP_distance(is_pip, P, perp_distance):
    """
    Input:
            P: input sequence
    Output:
            returns a point with maximum distance to P[1] and P[-1]
    """

    for i in range(1, len(P)):
        if is_pip[i]:
            perp_distance[i] = -1
        else:
            index_left = get_adjacent_pip_index(i, is_pip, side="left")
            index_right = get_adjacent_pip_index(i, is_pip, side="right")

            P1 = [index_left, P[index_left]]
            P2 = [index_right, P[index_right]]

            distance_P1_P2 = ((P2[1] - P1[0]) ** 2 +
                              (P2[0] - P1[0]) ** 2) ** 0.5

            perp_distance[i] = perpendicular_distance(
                P1, P2, [i, P[i]], distance_P1_P2)

    index_max = np.argmax(perp_distance)

    return perp_distance, index_max


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


def inverse_head_and_shoulder_rule(SP, diff_value=0.15):
    """
    Input:
        SP: 7 point PIP identified points
        diff_value: maximum permissible difference between 2 values
    Output:
        returns a boolean if SP points satisfy IHS pattern
    Description:
        SP starts from index 0
        sp3 < sp1 and sp5
        sp1 < sp0 and sp2
        sp5 < sp4 and sp6
        sp2 < sp0
        sp4 < sp6
        diff(sp1, sp5) < diff_value
        diff(sp2, sp4) < diff_value
    """
    if not SP:
        return False

    if SP[3] > SP[1] or SP[3] > SP[5] or SP[1] > SP[0] or SP[1] > SP[2] or SP[5] > SP[4] or SP[5] > SP[6] or SP[2] > SP[0] or SP[4] > SP[6]:
        return False

    if abs((SP[1] - SP[5]) * 1.0 / min(SP[1], SP[5])) >= diff_value:
        return False

    if abs((SP[2] - SP[4]) * 1.0 / min(SP[2], SP[4])) >= diff_value:
        return False

    return True


def template_matching(PIP, PIP_time, template, template_time):
    """
    Input:
        PIP: Input sequence for pattern to be matched against.
        template: Input sequence for pattern.
    Output:
        Returns the distortion value, of which a value closer
        to 0 will represent a better match.
    """

    # Lengths must be the same for them to match.
    if ((len(PIP) != len(template)) or (len(PIP_time) != len(template)) or
            len(template_time) != len(template)):
        return np.inf

    N = len(PIP)

    PIP = np.array(PIP)
    PIP_time = np.array(PIP_time)
    template = np.array(template)
    template_time = np.array(template_time)

    # Normalize all points to between 0 and 1
    PIP = PIP / np.abs(PIP).max()
    template = template / np.abs(template).max()

    # Normalize the time data points between 0 and 1
    template_time = template_time - template_time[0]
    PIP_time = PIP_time - PIP_time[0]
    template_time = template_time / template_time[-1]
    PIP_time = PIP_time / PIP_time[-1]

    # Amplitude Distance - the y-axis difference
    AD = np.linalg.norm(template - PIP) / np.sqrt(N)

    # Temporal Distance - the x-axis difference
    TD = np.linalg.norm(template_time - PIP_time) / np.sqrt(N - 1)

    distortion = AD * TEMPLATE_OMEGA_WT + TD * (1 - TEMPLATE_OMEGA_WT)

    return distortion


def temporal_control_penalty(slen, dlen, dlc):
    """
    Input:
        slen: Subsequence length.
        dlen: Desired length of matching subsequence
        dlc: Desired length control parameter. Controls sharpness of curve.
    Output:
        Penalty value
    Description:
        Calculates the penalty value for the matching subsequence vs the pattern.
    """

    theta = dlen / dlc
    d = slen - dlen
    tc = 1 - np.exp(-((d / theta)**2))

    return tc
