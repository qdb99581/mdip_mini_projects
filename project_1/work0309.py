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

print(y_1, y_2, y_3)

# L2 Distance
dist_12 = np.linalg.norm(x_1 - x_2)
dist_13 = np.linalg.norm(x_1 - x_3)
print(
    f"The euclidean distance between pixel 1 and pixel 2: {dist_12:.4f} (different classes)")
print(
    f"The euclidean distance between pixel 1 and pixel 3: {dist_13:.4f} (same classes)")
print("-"*50)
# Spectral Angles Mappers
val_12 = MC.SAM(x_1, x_2)
val_13 = MC.SAM(x_1, x_3)
print(
    f"The SAM value between pixel 1 and pixel 2: {val_12:.4f} (different classes)")
print(
    f"The SAM value between pixel 1 and pixel 3: {val_13:.4f} (same classes)")

# Spectral Information Divergence (SID)
SID_F = 0
SID_R = 0
for i in range(200):
    x_1 = int(x_data[30, 30, i])
    x_2 = int(x_data[120, 120, i])
    SID_F += MC.KLD(x_1, x_2)
    SID_R += MC.KLD(x_2, x_1)
print(SID_F+SID_R)


# plt.figure()
# plt.imshow(Indian_corrected[:, :, 90])
# plt.show()
