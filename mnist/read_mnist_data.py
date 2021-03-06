# copied from https://gist.github.com/akesling/5358964

import os
import struct

import matplotlib.pyplot as plt
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def read(dataset="training", path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'training', 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'training', 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 'testing', 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'testing', 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("data set must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    return {
        'feature_vectors': img,
        'class_labels': lbl
    }


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=plt.get_cmap('gray'))
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


# Everything that follows is my own code


def get_mnist_data_for_numbers(numbers):
    mnist = read('training', '..\\..\\mnist')
    feature_vectors = []
    class_labels = []
    for i, fv in enumerate(mnist['feature_vectors']):
        if mnist['class_labels'][i] in numbers:
            feature_vectors.append(fv)
            class_labels.append(mnist['class_labels'][i])
    return {
        'feature_vectors': np.array(feature_vectors),
        'class_labels': np.array(class_labels)
    }
