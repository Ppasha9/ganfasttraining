"""
Script for loading generator model from file and sample random images from codes.

Usage:
    sample_images_script.py (--model-path=PATH) (--pics-num=NUM_OF_PICS) (--pics-size=SIZE) (--output-dir=PATH)
    sample_images_script.py --help

Options:
    -m, --model-path=PATH           Full path to the saved KERAS model of generator.
    -n, --pics-num=NUM_OF_PICS      Number of result pictures
    -s, --pics-size=SIZE            Number of sampled pictures in one row/col
    -o, --output-dir=PATH           Full path to the directory, where result pictures will be.
    -h, --help                      Show this message
"""

import os
import numpy
import docopt

import matplotlib.pyplot as plt

from keras.models import load_model, Sequential
from keras.layers import Dense, Reshape, Conv2D, UpSampling2D, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU


def _get_generator_model(latent_dim):
        model = Sequential()

        model.add(Dense(1024, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(128 * 7 * 7))
        model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(1, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        return model


def _run(opts):
    model_path = opts['--model-path']
    if not os.path.exists(model_path):
        print("Full path to the model is invalid: %s" % model_path)
        return

    output_dir = opts['--output-dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    latent_dim = 100
    model = _get_generator_model(latent_dim)
    model.load_weights(model_path)
    

    pics_num = opts['--pics-num']
    pics_in_one_row = opts['--pics-size']
    for i in range(pics_num):
        noise = numpy.random.normal(size=(pics_in_one_row * pics_in_one_row, latent_dim))
        gen_imgs = model.predict(noise)

        # rescale images to 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, ax = plt.subplots(pics_in_one_row, pics_in_one_row, sharex=True, sharey=True, figsize=(15, 15))
        cnt = 0

        for j in range(pics_in_one_row):
            for k in range(pics_in_one_row):
                ax[j, k].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                ax[j, k].axis('off')
                cnt += 1

        fig.savefig(os.path.join(output_dir, "images_from_random_codes_" + str(i) + ".png"))
        plt.close()


if __name__ == "__main__":
    _run(docopt.docopt(__doc__))
