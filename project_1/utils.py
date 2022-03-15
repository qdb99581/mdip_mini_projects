import math
import numpy as np

def pixel_SAM(pixel_a, pixel_b):
    denominator_a = math.sqrt(np.sum(np.power(pixel_a, 2)))
    denominator_b = math.sqrt(np.sum(np.power(pixel_b, 2)))

    denominator = denominator_a * denominator_b
    nominator = np.sum(np.dot(pixel_a, pixel_b))
    
    sam = math.acos(nominator / denominator)
    
    return sam
