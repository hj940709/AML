__author__ = 'sorkhei'

import numpy as np
import struct

import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def read_mnist_training_data(N=60000):
    """
    :param N: the number of digits to be read, default is value is set to maximum = 60000
    :return: a list of tuples (X, y). X is a 28 by 28 image and y the corresponding label, a number between 0 and 9
    """
    files = os.listdir(os.getcwd())
    if 'train-images-idx3-ubyte' not in files or 'train-labels-idx1-ubyte' not in files:
        exit('training data not found')
    train_image = open('train-images-idx3-ubyte', 'rb')
    train_label = open('train-labels-idx1-ubyte', 'rb')
    _, _ = struct.unpack('>II', train_label.read(8))
    labels = np.fromfile(train_label, dtype=np.int8)
    _, _, img_row, img_col = struct.unpack('>IIII', train_image.read(16))
    images = np.fromfile(train_image, dtype=np.uint8).reshape(len(labels), img_row * img_col)
    return images[0:N, :], labels[0:N]


def visualize(image):
    """
    :param image: is a 28 by 28 image or a vector of images
    """
    if image.ndim == 1:
        image = np.array([image])
    cols = int(np.ceil(np.sqrt(image.shape[0])))
    img_number = 0
    for row in range(0, cols):
        for col in range(0, cols):
            if img_number > image.shape[0] - 1:
                break
            else:
                ax = plt.subplot2grid((cols, cols), (row, col))
                ax.axes.axes.get_xaxis().set_visible(False)
                ax.axes.axes.get_yaxis().set_visible(False)
                imgplot = ax.imshow(image[img_number].reshape(28, 28), cmap=mpl.cm.Greys)
                imgplot.set_interpolation('nearest')
                ax.xaxis.set_ticks_position('top')
                ax.yaxis.set_ticks_position('left')
                img_number += 1
    plt.show()




