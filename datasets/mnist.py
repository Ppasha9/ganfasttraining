import numpy

from keras.datasets import mnist


def get_dataset_rescaled():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train.astype(numpy.float32) - 127.5) / 127.5
    x_train = numpy.expand_dims(x_train, axis=3)

    x_test = (x_test.astype(numpy.float32) - 127.5) / 127.5
    x_test = numpy.expand_dims(x_test, axis=3)

    return x_train, y_train, x_test, y_test
