#!/bin/python3
import numpy as np

TEMPLATE_OMEGA_WT = 0.4


def PIP_identification(P, P_time, Q_length=7, isNumpyFormat=False):
    """
    Input:
            P_time: time sequence
            P: input sequence
            Q_length: number of PIPs
    Output:
            returns PIPs
    """
    N = len(P)
    if N == 0:
        print("Length cannot be zero!")

    if N < Q_length:
        return [], []


    # Converted PIP identification to Numpy, for performance
    if isNumpyFormat == False:
        P = np.array(P)
        P_time = np.array(P_time)
    pip_indexes = [0, N-1]

    for i in range(1, Q_length - 1):
        index = PIP_distance(pip_indexes, P)
        pip_indexes.append(index)
        pip_indexes.sort()

    SP = P[pip_indexes]
    SP_time = P_time[pip_indexes]

    return SP, SP_time


def get_adjacent_pip_index(index, is_pip, side):
    """
    Input:
            is_pip: Indicator if a particular value has been identified as PIP
            index: Current index
            side: side to which we need the nearest PIP
    Output:
            returns nearest PIP
    """

    k = index
    if side == "right":
        while not is_pip[k]:
            k += 1
    if side == "left":
        while not is_pip[k]:
            k -= 1
    return k


def PIP_distance(pip_indexes, P):
    """
    Input:
            is_pip: Indicator if a particular value has been identified as PIP
            P: input sequence
    Output:
            returns a point with maximum distance to nearest PIPs
    """
    N = len(P)

    P1 = np.zeros((N,2))
    P2 = np.zeros((N,2))
    distance_P1_P2 = np.full(N,np.inf) # Fill with inf value

    for i in range(len(pip_indexes) - 1):
        idx = int(pip_indexes[i])
        idx_right = int(pip_indexes[i+1])

        P1[idx+1:idx_right,0] = idx
        P2[idx+1:idx_right,0] = idx_right

        P1[idx+1:idx_right,1] = P[idx]
        P2[idx+1:idx_right,1] = P[idx_right]

        distance_P1_P2[idx+1:idx_right] = np.sqrt(((P[idx_right] - P[idx]) ** 2) +
                                                     ((idx_right - idx) ** 2))


    P3 = np.stack((np.arange(N),P),axis=1)

    perp_distance = perpendicular_distance(
        P1, P2, P3, distance_P1_P2)

    index_max = np.nanargmax(perp_distance)

    return index_max


def perpendicular_distance(P1, P2, P3, distance):
    """
    Input:
            P1,P2,P3: 3 points in [x,y] format
            S: slope between P1 and P2; this can pre-calculated
    Output:
            returns (perpendicular distance) between these 3 points
    """

    PD = np.abs((P2[:,1] - P1[:,1]) * P3[:,0] - (P2[:,0] - P1[:,0]) *
                 P3[:,1] + P2[:,0] * P1[:,1] - P2[:,1] * P1[:,0]) / distance[:]

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

    N = len(template)

    if ((len(PIP) < N) or (len(PIP_time) < N)):
        return np.inf

    # Lengths must be the same for them to match.
    if len(PIP) != N:
        PIP = np.array(PIP[:N])
        PIP_time = np.array(PIP_time[:N])

    #template = np.array(template)
    #template_time = np.array(template_time)

    # Normalize all points to between 0 and 1
    PIP = PIP / np.abs(PIP).max()
    template = template / np.abs(template).max()

    # Normalize the time data points between 0 and 1
    template_time = template_time - template_time[0]
    PIP_time = PIP_time - PIP_time[0]
    template_time = template_time / template_time[-1]
    PIP_time = PIP_time / PIP_time[-1]

    # Amplitude Distance - the y-axis difference
    #PIP_inverse = 1 - PIP
    #PIP_normalize = np.maximum(PIP,PIP_inverse)
    #AD = np.linalg.norm((template - PIP)/PIP_normalize) / np.sqrt(N)
    AD = np.linalg.norm(template - PIP) / np.sqrt(N)

    # Temporal Distance - the x-axis difference
    TD = np.linalg.norm(template_time - PIP_time) / np.sqrt(N - 1)

    distortion = AD * TEMPLATE_OMEGA_WT + TD * (1 - TEMPLATE_OMEGA_WT)

    return distortion


def multiple_template_matching(PIP, PIP_time, template_list):
    """
    Input:
        PIP: PIP values, y-axis
        PIP_time: PIP values, x-axis(time)
        template_list: Dict of templates, with format
                       template['template_name']['x or y']

    Output: Tuple containing the minimum distortion value and the corresponding
            pattern name (distortion_val,pattern_name)
    Description:
        Match against mutiple templates, and return the template with the lowest
        distortion.
    """

    distortion_min = np.inf
    min_pattern_name = ''

    for template_name, template_data in template_list.items():
        val = template_matching(PIP,PIP_time,
                                   template_data['y'],template_data['x'])
        #print('Distortion (' + template_name + '): ' + str(val))

        if val < distortion_min:
            distortion_min = val
            min_pattern_name = template_name

    return (distortion_min,min_pattern_name)







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


