import numpy

from keras.datasets import mnist


def get_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = numpy.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = numpy.reshape(x_test, (len(x_test), 28, 28, 1))

    return x_train, y_train, x_test, y_test
