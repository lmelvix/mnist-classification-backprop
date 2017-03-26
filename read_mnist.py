import os
import struct
from array import array as pyarray
from pylab import *
from numpy import *
from numpy import array, int8, uint8, zeros
import warnings


def load_mnist(dataset="training", digits=np.arange(10), path=""):
    '''
    Load Binary MNIST dataset images and corresponding labels
    :param dataset: Type of dataset to be loaded
    :param digits:  Range of digits to be stored
    :param path:  Path to MNIST dataset
    :return: Images and corresponding labels
    '''
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
    images = zeros((N, rows * cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)

    for i in range(len(ind)):
            images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
            labels[i] = lbl[ind[i]]

    return images, labels


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as plt
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=plt.cm.gray)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()