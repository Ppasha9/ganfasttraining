import sys
import numpy
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import pydot

from keras.layers import Input, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Reshape, Lambda
from keras.layers import concatenate
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.utils import model_to_dot
from keras.utils.vis_utils import plot_model

from IPython.display import SVG

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = numpy.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = numpy.reshape(x_test,  (len(x_test),  28, 28, 1))

y_train_cat = to_categorical(y_train).astype(numpy.float32)
y_test_cat  = to_categorical(y_test).astype(numpy.float32)

print(y_train_cat.shape)

# Регистрация сессии в keras
from keras import backend as K
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)

batch_size = 500
latent_dim = 100
dropout_rate = 0.3
start_lr = 0.001
num_classes = 10


def create_cvae():
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_rate)(BatchNormalization()(x))

    input_img = Input(shape=(28, 28, 1))
    flatten_img = Flatten()(input_img)
    input_lbl = Input(shape=(num_classes,), dtype='float32')

    with tf.variable_scope('encoder_model'):
        x = concatenate([flatten_img, input_lbl])
        x = Dense(256, activation='relu')(x)
        x = LeakyReLU()(x)
        x = apply_bn_and_dropout(x)
        x = Dense(128, activation='relu')(x)
        x = LeakyReLU()(x)
        x = apply_bn_and_dropout(x)

        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder_model = Model([input_img, input_lbl], l, name="Encoder")
    models["encoder"] = encoder_model

    z = Input(shape=(latent_dim,))
    input_lbl_d = Input(shape=(num_classes,), dtype='float32')

    with tf.variable_scope('decoder_model'):
        x = concatenate([z, input_lbl_d])
        x = Dense(256)(x)
        x = LeakyReLU()(x)
        x = apply_bn_and_dropout(x)
        x = Dense(28 * 28, activation='sigmoid')(x)
        decoded = Reshape((28, 28, 1))(x)
    decoder_model = Model([z, input_lbl_d], decoded, name='Decoder')
    models["decoder"] = decoder_model
    models["cvae"] = Model([input_img, input_lbl, input_lbl_d],
                           models["decoder"]([models["encoder"]([input_img, input_lbl]), input_lbl_d]),
                           name="CVAE")

    def vae_loss(x, decoded):
        x = K.reshape(x, shape=(batch_size, 28 * 28))
        decoded = K.reshape(decoded, shape=(batch_size, 28 * 28))
        xent_loss = 28 * 28 * binary_crossentropy(x, decoded)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return (xent_loss + kl_loss) / 2 / 28 / 28

    return models, vae_loss


models, vae_loss = create_cvae()
cvae = models["cvae"]
encoder = models["encoder"]
decoder_model = models["decoder"]

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/User/Downloads/graphviz-2.38/release/bin/'
plot_model(encoder, to_file="saved_models\\conditional_variational_ae\\encoder.png")
plot_model(decoder_model, to_file="saved_models\\conditional_variational_ae\\decoder.png")