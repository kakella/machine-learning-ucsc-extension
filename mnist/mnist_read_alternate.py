import os
from struct import unpack

from numpy import zeros, uint8, float32


def get_labeled_data(dataset, path):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'training', 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'training', 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 'testing', 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'testing', 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("data set must be 'testing' or 'training'")

    images = open(fname_img, 'rb')
    labels = open(fname_lbl, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return x, y


def read_mnist():
    return get_labeled_data('training', '..\\..\\mnist')
