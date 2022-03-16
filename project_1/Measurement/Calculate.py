import math

import numpy as np


# Square Difference Formula (preprocessing part)
def SDF(rx, ry):
    return(rx-ry)**2


def SAM(pixel_1, pixel_2):
    """Calculate spectral angles mapper (SAM) for two pixels.

    Args:
        pixel_1 (ndarray): [n_bands, 1]
        pixel_2 (ndarray): [n_bands, 1]

    Returns:
        float: SAM value
    """
    numerator = 0
    denominator_l = 0
    denominator_r = 0

    n_bands = len(pixel_1)
    for i in range(n_bands):
        rx, ry = float(pixel_1[i]), float(pixel_2[i])

        numerator += rx * ry
        denominator_l += rx * rx
        denominator_r += ry * ry

    res = math.acos(numerator / (math.sqrt(denominator_l)
                    * math.sqrt(denominator_r)))
                    
    return res * 180 / math.pi


def SID(pixel_1, pixel_2):
    """Calculate spectral information divergence (SID)

    Args:
        pixel_1 (ndarray): [n_bands, 1]
        pixel_2 (ndarray): [n_bands, 1]

    Returns:
        float: SID value
    """
    SID_F = 0
    SID_R = 0

    n_bands = len(pixel_1)
    for i in range(n_bands):
        rx, ry = float(pixel_1[i]), float(pixel_2[i])
        SID_F += _KLD(rx, ry)
        SID_R += _KLD(ry, rx)

    return SID_F + SID_R


# Kullback-Leibler divergence (KLD) (preprocessing part)
def _KLD(rx, ry):
    return(rx*np.log(rx/ry))
