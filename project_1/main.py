from scipy.io import loadmat

if __name__ == "__main__":
    data_dir = './dataset/Indian_pines_corrected.mat'
    label_dir = "dataset/Indian_pines_gt.mat"
    x_data = loadmat(data_dir)["indian_pines_corrected"]
    y_data = loadmat(label_dir)["indian_pines_gt"]
    print(x_data.shape)
    print(y_data.shape)