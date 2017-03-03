import glob

import cv2
import numpy as np
from skimage import feature as sk_feature
from skimage import io
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
    if vis:
        features, hog_image = sk_feature.hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                             cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                             visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = sk_feature.hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=False, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def featurize(img, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(16, 16), hist_bins=32):
    assert (img.shape == (64, 64, 3))
    assert (img.dtype == np.uint8)

    img = img.astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    assert (img.shape[2] == 3)
    hog1 = get_hog_features(img[:, :, 0], orient, pix_per_cell, cell_per_block, feature_vec=True)
    hog2 = get_hog_features(img[:, :, 1], orient, pix_per_cell, cell_per_block, feature_vec=True)
    hog3 = get_hog_features(img[:, :, 2], orient, pix_per_cell, cell_per_block, feature_vec=True)
    hog_features = np.hstack((hog1, hog2, hog3))

    spatial_features = bin_spatial(img, size=spatial_size)
    hist_features = color_hist(img, nbins=hist_bins)

    return np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)


def read_images(data_path='./data/vehicles', func=None):
    """
    Read all images in data_path, and apply the function func on each image.
    func has to be a function of 1 argument (the image array), and returns a feature vector for that image
    """
    feats = []
    for p in sorted(glob.glob('{}/*/*.png'.format(data_path))):
        feat = func(io.imread(p))
        feats.append(feat[0, :])
    return np.asarray(feats)


def main():
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (16, 16)
    hist_bins = 32

    def featurize_func(img):
        return featurize(img, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    vehicles = read_images('./data/vehicles', featurize_func)
    non_vehicles = read_images('./data/non-vehicles', featurize_func)

    print(vehicles.shape)
    print(non_vehicles.shape)
    np.savez('./data/features.npz', vehicles=vehicles, non_vehicles=non_vehicles)

    X = np.vstack((vehicles, non_vehicles)).astype(np.float64)
    y = np.hstack((np.ones(len(vehicles)), np.zeros(len(non_vehicles))))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ppl = Pipeline([('scaler', StandardScaler()), ('svc', LinearSVC())])

    ppl.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', round(ppl.score(X_test, y_test), 4))
    persist_data = {'ppl': ppl,
                    'orient': orient,
                    'pix_per_cell': pix_per_cell,
                    'cell_per_block': cell_per_block,
                    'spatial_size': spatial_size,
                    'hist_bins': hist_bins}
    joblib.dump(persist_data, 'car_model.pkl')


if __name__ == '__main__':
    main()
