from scipy.io import loadmat

import utils

if __name__ == "__main__":
    data_dir = './dataset/Indian_pines_corrected.mat'
    label_dir = './dataset/Indian_pines_gt.mat'

    x_data = loadmat(data_dir)["indian_pines_corrected"]
    y_data = loadmat(label_dir)["indian_pines_gt"]

    # Pixel A: Class = 3, at [0][0]
    # Pixel B: Class = 3, at [0][1]
    # Pixel C: Class = 0, at [0][144]
    pixels = {
        "Pixel A": (x_data[50][50], y_data[50][50]),
        "Pixel B": (x_data[80][80], y_data[80][80]),
        "Pixel C": (x_data[0][144], y_data[0][144]),
    }

    sam = utils.pixel_SAM(pixels["Pixel A"][0], pixels["Pixel B"][0])
    print(sam)