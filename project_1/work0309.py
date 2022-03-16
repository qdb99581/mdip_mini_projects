import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Measurement.Calculate as MC
import math


# Load data: [145, 145, 200]
x_data = sio.loadmat(
    "./dataset/Indian_pines_corrected.mat")['indian_pines_corrected']
y_data = sio.loadmat("./dataset/Indian_pines_gt.mat")['indian_pines_gt']

x_1 = x_data[50, 50, :]  # Class 0
x_2 = x_data[80, 80, :]  # Class 2
x_3 = x_data[51, 51, :]  # Class 0

y_1 = y_data[50, 50]
y_2 = y_data[80, 80]
y_3 = y_data[51, 51]

print(f"Class of pixel 1: {y_1}")
print(f"Class of pixel 2: {y_2}")
print(f"Class of pixel 3: {y_3}")
print("-"*50)

# L2 Distance
dist_12 = np.linalg.norm(x_1 - x_2)
dist_13 = np.linalg.norm(x_1 - x_3)
print(
    f"The euclidean distance between pixel 1 and pixel 2: {dist_12:.4f} (different classes)")
print(
    f"The euclidean distance between pixel 1 and pixel 3: {dist_13:.4f} (same classes)")
print("-"*50)

# Spectral Angles Mappers
sam_12 = MC.SAM(x_1, x_2)
sam_13 = MC.SAM(x_1, x_3)
print(
    f"The SAM value between pixel 1 and pixel 2: {sam_12:.4f} (different classes)")
print(
    f"The SAM value between pixel 1 and pixel 3: {sam_13:.4f} (same classes)")
print("-"*50)

# Spectral Information Divergence (SID)
sid_12 = MC.SID(x_1, x_2)
sid_13 = MC.SID(x_1, x_3)
print(
    f"The SID value between pixel 1 and pixel 2: {sid_12:.4f} (different classes)")
print(
    f"The SID value between pixel 1 and pixel 3: {sid_13:.4f} (same classes)")


# plt.figure()
# plt.imshow(Indian_corrected[:, :, 90])
# plt.show()
