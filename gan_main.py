import os
import json
import time
import numpy

import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K

from keras.optimizers import Adam
from keras.datasets import mnist

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input
from keras.layers.advanced_activations import LeakyReLU


if __name__ == "__main__":
    models = {}

    save_path = os.path.join("saved_models", "conditional_variational_ae")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # with open(os.path.join(save_path, "cvae.json"), "r") as f:
    #     data = f.read()
    # models["cvae"] = model_from_json(data)
    # models["cvae"].load_weights(os.path.join(save_path, "cvae.h5"))

    with open(os.path.join(save_path, "cvae_encoder.json"), "r") as f:
        data = f.read()
    models["encoder"] = model_from_json(data)
    models["encoder"].load_weights(os.path.join(save_path, "cvae_encoder.h5"))

    with open(os.path.join(save_path, "cvae_decoder.json"), "r") as f:
        data = f.read()
    models["decoder"] = model_from_json(data)
    models["decoder"].load_weights(os.path.join(save_path, "cvae_decoder.h5"))

    # Preparation
    numpy.random.seed(30)

    noise_dim = 100
    batch_size = 16
    steps_per_epoch = 3750
    epochs = 60

    save_path = "fcgan-images"

    img_rows, img_cols, channels = 28, 28, 1

    optimizer = Adam(0.0002, 0.5)

    # Data preparation
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(numpy.float32) - 127.5) / 127.5

    x_train = x_train.reshape(-1, img_rows * img_cols * channels)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Generator
    def _create_generator():
        generator = Sequential()

        generator.add(Dense(256, input_dim=noise_dim))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(img_rows * img_cols * channels, activation='tanh'))

        generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return generator

    # Discriminator
    def _create_discriminator():
        discriminator = Sequential()

        discriminator.add(Dense(1024, input_dim=img_rows * img_cols * channels))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Dense(1, activation='sigmoid'))

        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return discriminator

    discriminator = _create_discriminator()
    generator = _create_generator()

    discriminator.trainable = False

    gan_input = Input(shape=(noise_dim,))
    fake_image = generator(gan_input)

    gan_output = discriminator(fake_image)

    gan = Model(gan_input, gan_output, name="Simple GAN Model")
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    gan.summary()

    start_time = time.time()
    for epoch in range(epochs):
        start_epoch_time = time.time()
        for batch in range(steps_per_epoch):
            noise = numpy.random.normal(0, 1, size=(batch_size, noise_dim))
            fake_x = generator.predict(noise)

            real_x = x_train[numpy.random.randint(0, x_train.shape[0], size=batch_size)]

            x = numpy.concatenate((real_x, fake_x))

            disc_y = numpy.zeros(2 * batch_size)
            disc_y[:batch_size] = 0.9

            d_loss = discriminator.train_on_batch(x, disc_y)

            gen_y = numpy.ones(batch_size)
            g_loss = gan.train_on_batch(noise, gen_y)

        print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}'
              f'\t\t Elapsed time for epoch: {time.time() - start_epoch_time}')
    print(f'\nELAPSED TIME FOR ALL TRAINING: {time.time() - start_time}')

    def _show_images(noise):
            generated_images = generator.predict(noise)
            plt.figure(figsize=(10, 10))

            for i, image in enumerate(generated_images):
                plt.subplot(10, 10, i + 1)
                if channels == 1:
                    plt.imshow(image.reshape((img_rows, img_cols)), cmap='gray')
                else:
                    plt.imshow(image.reshape((img_rows, img_cols, channels)))
                plt.axis('off')

            plt.tight_layout()
            plt.show()

    noise = numpy.random.normal(0, 1, size=(100, noise_dim))
    _show_images(noise)

    save_path = os.path.join("saved_models", "simple_gan")
    with open(os.path.join(save_path, "gan_model.json"), "w") as f:
        json.dump(gan.to_json(), f, indent=4)
    gan.save_weights(os.path.join(save_path, "gan_model.h5"))

    with open(os.path.join(save_path, "gan_gen_model.json"), "w") as f:
        json.dump(generator.to_json(), f, indent=4)
    generator.save_weights(os.path.join(save_path, "gan_gen_model.h5"))

    with open(os.path.join(save_path, "gan_discr_model.json"), "w") as f:
        json.dump(discriminator.to_json(), f, indent=4)
    discriminator.save_weights(os.path.join(save_path, "gan_discr_model.h5"))
