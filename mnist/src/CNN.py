# CNN.py
# Explore implementation of a Convolutional NN

# TODO LIST:
# -- amend preprocessing to handle CNN (i.e. do not flatten)
# -- implement single 3x3 filter and apply to a MNIST image, view result
# -- build out a CNN layer object / forward pass
# -- build out backprop for CNN
# -- consider padding/stride?

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from src.data import get_dataset, preprocess
from src.vis import draw_grid, draw_1

def test_filter():
    x_raw, y_raw, x_raw_test, y_raw_test = get_dataset() # get train/test data
    x_train, y_train = preprocess(x_raw, y_raw, flatten=False)

    
    # create a "vertical line" filter as example
    F_vertical = np.array([
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0]
    ], dtype=np.float32)
    # create a "horizontal line" filter as example
    F_horizontal = np.array([
        [-1.0, -1.0, -1.0],
        [+0.0, +0.0, +0.0],
        [+1.0, +1.0, +1.0]
    ], dtype=np.float32)

    # apply filter to first image in training set, ignoring first dim. ("channel")
    def conf2Dtest(image, F):
        Fh, Fw = F.shape # filter height and width
        gap = int(Fh / 2.0) # "size" of the filter
        img = image[gap:-gap, gap:-gap] # the pixels where we can apply the filter
        height, width = image.shape
        result = np.zeros_like(img)
        for h in range(gap, height-gap, 1):
            for w in range(gap, width-gap, 1):
                window = image[h-gap:h+gap+1,w-gap:w+gap+1]
                # apply the filter for this (h,w) location
                result[h-gap,w-gap] = np.sum( window * F )
        return result
    r1 = conf2Dtest(x_train[0][0], F_vertical)
    r2 = conf2Dtest(x_train[0][0], F_horizontal)

    # [before, after]
    gap = 1
    img = x_train[0][0][gap:-gap, gap:-gap]
    draw_grid(np.array([img, r1, r2]), np.array([y_raw[0]]*3), np.array([0,1,2]), (1,3), scale=3)
    #draw_1(np.array([img, result]), [y_raw[0], y_raw[0]], 0, scale=3)

#==============================================================================

if __name__ == "__main__":
    test_filter()
