import os
import json
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wgan_gp.wgan_gp_settings as settings

from tensorflow_gan.examples.mnist import util

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Activation, Dropout
from keras.layers.merge import _Merge
from keras.layers.convolutional import Conv2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from keras import backend as K

from functools import partial
from PIL import Image

from datasets import get_mnist_dataset_rescaled
from logger import wgan_gp_logger


np.random.seed(30)


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((settings.BATCH_SIZE, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gp_loss(y_true, y_pred, averaged_samples, gp_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradients_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gp_weight * K.square(1 - gradients_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


class WGAN_GP_MODEL_FOR_MNIST(object):
    def __init__(self):
        self.batch_size = settings.BATCH_SIZE
        self.pics_output_dir = settings.OUTPUT_DIR
        self.checkpoint_dir = settings.CHECKPOINT_DIR
        self.critic_iters = settings.CRITIC_UPDATES_ITERS
        self.gp_weight = settings.GP_WEIGHT
        self.latent_dim = settings.LATENT_DIM

        self.sample_interval = settings.SAMPLE_INTERVAL
        self.checkpoint_interval = settings.CHECKPOINT_INTERVAL

        self.x_train = list()
        self.x_fid_test = list()

        self.cur_fid_dist = None

        self.need_break = False
        self.num_of_good_epochs = 0
        self.start_train_time = 0

        self.critic_loss_list = list()
        self.generator_loss_list = list()
        self.fid_score_list = list()

        self.img_rows = settings.IMG_ROWS
        self.img_cols = settings.IMG_COLS
        self.img_channels = settings.IMG_CHANNELS
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        optimizer = RMSprop(lr=0.00005)

        self.generator = self._make_generator()
        self.critic = self._make_critic()

        # ===== Construct Computational Graph for the Critic =====

        # freeze generator layers while training the critic
        for layer in self.generator.layers:
            layer.trainable = False
        self.generator.trainable = False

        # real image sample
        real_img_sample = Input(shape=self.img_shape)

        # Noise input for critic
        z_noise_critic = Input(shape=(self.latent_dim,))
        # fake image sample
        fake_img_sample = self.generator(z_noise_critic)

        # Critic determines validity of real and fake images
        fake_validity = self.critic(fake_img_sample)
        real_validity = self.critic(real_img_sample)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img_sample, fake_img_sample])
        # Determine validity of weighted sample
        weighted_validity = self.critic(interpolated_img)

        partial_gp_loss = partial(gp_loss, averaged_samples=interpolated_img, gp_weight=self.gp_weight)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model([real_img_sample, z_noise_critic], [real_validity, fake_validity, weighted_validity], name="WGAN_GP_Critic")
        self.critic_model.compile(optimizer=optimizer,
                                  loss_weights=[1, 1, 10],
                                  loss=[wasserstein_loss,
                                        wasserstein_loss,
                                        partial_gp_loss])

        # ===== Construct Computational Graph for the Generator =====

        # freeze critic's layers
        for layer in self.critic.layers:
            layer.trainable = False
        self.critic.trainable = False
        for layer in self.generator.layers:
            layer.trainable = True
        self.generator.trainable = True

        # Noise input for generator
        z_noise_gen = Input(shape=(self.latent_dim,))
        # Generated images
        generated_img = self.generator(z_noise_gen)
        # Critic determines validity of generated images
        gen_img_validity = self.critic(generated_img)

        self.generator_model = Model(z_noise_gen, gen_img_validity, name="WGAN_GP_Generator")
        self.generator_model.compile(optimizer=optimizer, loss=wasserstein_loss)

    def _make_generator(self):
        model = Sequential()

        model.add(Dense(1024, input_dim=self.latent_dim))
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

        model.add(Conv2D(self.img_channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise_input = Input(shape=(self.latent_dim,))
        generated_img = model(noise_input)

        return Model(noise_input, generated_img)

    def _make_critic(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        critic_input_img = Input(shape=self.img_shape)
        validity = model(critic_input_img)

        return Model(critic_input_img, validity)

    def train(self):
        # load dataset
        x_train, _, x_test, _ = get_mnist_dataset_rescaled()

        self.x_train = x_train
        self.x_fid_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

        # adversarial ground truth
        valid = -np.ones((self.batch_size, 1), dtype=np.float32)
        fake = np.ones((self.batch_size, 1), dtype=np.float32)
        dummy = np.zeros((self.batch_size, 1), dtype=np.float32)   # dummy for gradient penalty

        minibatches_size = self.batch_size * self.critic_iters
        n_batches = x_train.shape[0] // minibatches_size

        epoch = 0
        self.start_train_time = time.time()
        wgan_gp_logger.info("==== START TRAINING ====")
        while not self.need_break:
            np.random.shuffle(self.x_train)
            for i in range(n_batches):
                critic_minibatches = self.x_train[i * minibatches_size: (i + 1) * minibatches_size]
                for j in range(self.critic_iters):
                    # ==== Train Critic ====

                    # random batch of images
                    image_batch = critic_minibatches[j * self.batch_size: (j + 1) * self.batch_size]
                    # generated noise
                    noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

                    # Train critic
                    try:
                        critic_loss = self.critic_model.train_on_batch([image_batch, noise], [valid, fake, dummy])
                    except Exception as e:
                        wgan_gp_logger.error("Exception while training critic: %s" % e)

                # ==== Train Generator ====
                try:
                    gen_loss = self.generator_model.train_on_batch(noise, valid)
                except Exception as e:
                    wgan_gp_logger.error("Exception while training generator: %s" % e)

                if i % 30 == 0:
                    wgan_gp_logger.info("[Epoch: %d] [Batch: %d/%d] [C loss: %f] [G loss: %f]"
                                        % (epoch, i, n_batches, critic_loss[0], gen_loss))
                self.critic_loss_list.append(critic_loss[0])
                self.generator_loss_list.append(gen_loss)

            if epoch % self.sample_interval == 0:
                wgan_gp_logger.info("Sample images on %d epoch" % epoch)
                self.sample_generator_images(epoch)

            if epoch % self.checkpoint_interval == 0:
                wgan_gp_logger.info("Save checkpoint on %d epoch" % epoch)
                self.save_checkpoint(epoch)

            epoch += 1

    def _gen_images_from_random_codes(self, r, c):
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # rescale images to 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, ax = plt.subplots(r, c, sharex=True, sharey=True, figsize=(15, 15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                ax[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                ax[i, j].axis('off')
                cnt += 1

        return fig

    def sample_generator_images(self, epoch):
        fig = self._gen_images_from_random_codes(10, 10)
        fig.savefig(os.path.join(self.pics_output_dir, "epoch_%d.png" % epoch))
        plt.close()

    def _calculate_current_fid_score(self):
        generated_images = self.generator.predict(np.random.normal(size=(len(self.x_fid_test), self.latent_dim)))
        generated_images = tf.convert_to_tensor(generated_images, dtype=tf.float32)

        return util.mnist_frechet_distance(self.x_fid_test, generated_images).numpy()

    def _save_sampled_images(self, dir_path):
        fig = self._gen_images_from_random_codes(10, 10)
        fig.savefig(os.path.join(dir_path, "images_from_random_codes.png"))
        plt.close()

    def _save_current_checkpoint_vars(self, dir_path):
        dict_to_save = dict()
        self.cur_fid_dist = self._calculate_current_fid_score()
        self.fid_score_list.append(self.cur_fid_dist)
        wgan_gp_logger.info("FID: %f" % self.cur_fid_dist)

        dict_to_save["fid"] = str(self.cur_fid_dist)
        dict_to_save["time_spended"] = str(time.time() - self.start_train_time)
        with open(os.path.join(dir_path, "vars.json"), "w") as f:
            json.dump(dict_to_save, f, indent=4)

        if self.cur_fid_dist <= 0.1:
            self.num_of_good_epochs += 1
            if self.num_of_good_epochs == 4:
                wgan_gp_logger.info("==== STOP TRAINING ====")
                self.need_break = True
        else:
            self.num_of_good_epochs = 0

    def _save_generator_model(self, dir_path):
        self.generator_model.save(filepath=os.path.join(dir_path, "generator_model.h5"))

    def _save_loss_funcs(self, dir_path):
        np.save(os.path.join(dir_path, "critic_loss.npy"), self.critic_loss_list)
        np.save(os.path.join(dir_path, "generator_loss.npy"), self.generator_loss_list)

        fig = plt.figure(figsize=(15, 15))
        plt.plot(np.asarray(self.critic_loss_list))
        fig.savefig(os.path.join(dir_path, "critic_loss.png"))
        plt.close()

        fig = plt.figure(figsize=(15, 15))
        plt.plot(np.asarray(self.generator_loss_list))
        fig.savefig(os.path.join(dir_path, "generator_loss.png"))
        plt.close()

    def _save_fid_score_list(self, dir_path):
        np.save(os.path.join(dir_path, "fid_score.npy"), self.fid_score_list)

        fig = plt.figure(figsize=(15, 15))
        plt.plot(np.asarray(self.fid_score_list))
        fig.savefig(os.path.join(dir_path, "fid_score.png"))
        plt.close()

    def save_checkpoint(self, epoch):
        dir_path = os.path.join(self.checkpoint_dir, "Epoch_%d" % epoch)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self._save_sampled_images(dir_path)
        self._save_current_checkpoint_vars(dir_path)
        self._save_generator_model(dir_path)
        self._save_loss_funcs(dir_path)
        self._save_fid_score_list(dir_path)
