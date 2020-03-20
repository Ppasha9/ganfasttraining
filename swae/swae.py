import os
import json
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import swae.swae_settings as settings

from tensorflow_gan.examples.mnist import util

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Activation, AveragePooling2D
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from keras import backend as K

from datasets import get_mnist_dataset_rescaled
from logger import swae_logger


class SWAE_MODEL(object):
    def __init__(self):
        self.batch_size = settings.BATCH_SIZE
        self.pics_output_dir = settings.OUTPUT_DIR
        self.checkpoint_dir = settings.CHECKPOINT_DIR
        self.gp_weight = settings.GP_WEIGHT
        self.latent_dim = settings.LATENT_DIM
        self.projections_num = settings.PROJECTIONS_NUM

        self.sample_interval = settings.SAMPLE_INTERVAL
        self.checkpoint_interval = settings.CHECKPOINT_INTERVAL

        self.x_train = list()
        self.x_fid_test = list()

        self.cur_fid_dist = None

        self.need_break = False
        self.start_train_time = 0

        self.loss_list = list()
        self.fid_score_list = list()

        self.img_rows = settings.IMG_ROWS
        self.img_cols = settings.IMG_COLS
        self.img_channels = settings.IMG_CHANNELS
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        optimizer = RMSprop(lr=0.00005)

        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()

        self.theta_var = K.variable(self._generate_theta())
        self.q_z_sample_var = K.variable(self._generate_qz())

        img = Input(shape=self.img_shape)
        img_encoded = self.encoder(img)
        z_decoded = self.decoder(img_encoded)
        self.swae_autoencoder = Model(img, z_decoded, name="SWAE")
        self.swae_autoencoder.summary()

        proj_encoded_samples = K.dot(img_encoded, K.transpose(self.theta_var))  # projection of the encoded samples
        proj_q_z = K.dot(self.q_z_sample_var, K.transpose(self.theta_var))           # projection of ideal q_z samples

        # Calculate Sliced Wasserstein distance by sorting the projections and calculating L2 distance
        sliced_wass_dist = (tf.nn.top_k(tf.transpose(proj_encoded_samples), k=self.batch_size).values -
                            tf.nn.top_k(tf.transpose(proj_q_z), k=self.batch_size).values) ** 2

        lambda_coeff = K.variable(self.gp_weight)

        cross_entropy_loss = (1.0) * K.mean(K.binary_crossentropy(K.flatten(img), K.flatten(z_decoded)))
        L1_loss = (1.0) * K.mean(K.abs(K.flatten(img) - K.flatten(z_decoded)))
        sliced_wass_loss = lambda_coeff * K.mean(sliced_wass_dist)

        swae_loss = L1_loss + cross_entropy_loss + sliced_wass_loss
        self.swae_autoencoder.add_loss(swae_loss)

        self.swae_autoencoder.compile(optimizer=optimizer, loss='')

    def _make_encoder(self):
        img = Input(shape=self.img_shape)
        x = Conv2D(16, (3, 3), padding='same')(img)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(16, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        encoded = Dense(self.latent_dim)(x)
        swae_encoder = Model(img, encoded, name="SWAE_Encoder")
        swae_encoder.summary()
        return swae_encoder

    def _make_decoder(self):
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

    def _generate_theta(self):
        res = [w / np.sqrt((w ** 2).sum()) for w in np.random.normal(size=(self.projections_num, self.latent_dim))]
        return np.asarray(res)

    def _generate_qz(self):
        return np.random.normal(size=(self.batch_size, self.latent_dim))

    def train(self):
        # load dataset
        x_train, _, x_test, _ = get_mnist_dataset_rescaled()

        self.x_train = x_train
        self.x_fid_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

        n_batches = int(self.x_train.shape[0] / self.batch_size)

        epoch = 0
        self.start_train_time = time.time()
        swae_logger.info("==== START TRAINING ====")
        while not self.need_break:
            ind = np.random.permutation(self.x_train.shape[0])

            for i in range(n_batches):
                next_batch = x_train[ind[i * self.batch_size: (i + 1) * self.batch_size], ...]

                theta_ = self._generate_theta()
                q_z_sample_ = self._generate_qz()
                K.set_value(self.theta_var, theta_)
                K.set_value(self.q_z_sample_var, q_z_sample_)

                try:
                    cur_loss = self.swae_autoencoder.train_on_batch(x=next_batch, y=None)
                except Exception as e:
                    swae_logger.error("Exception while training autoencoder: %s" % e)
                    return

                self.loss_list.append(cur_loss)

                if i % 30 == 0:
                    swae_logger.info("[Epoch: %d] [Batch: %d/%d] [Loss: %f]" % (epoch, i, n_batches, cur_loss))

            if epoch % self.sample_interval == 0:
                swae_logger.info("Sample images on %d epoch" % epoch)
                self._sample_decoder_images(epoch)

            if epoch % self.checkpoint_interval == 0:
                swae_logger.info("Save checkpoint on %d epoch" % epoch)
                self._save_checkpoint(epoch)

            epoch += 1

    def _gen_images_from_random_codes(self, r, c):
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.decoder.predict(noise)

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

    def _sample_decoder_images(self, epoch):
        fig = self._gen_images_from_random_codes(10, 10)
        fig.savefig(os.path.join(self.pics_output_dir, "epoch_%d.png" % epoch))
        plt.close()

    def _calculate_current_fid_score(self):
        generated_images = self.decoder.predict(np.random.normal(size=(len(self.x_fid_test), self.latent_dim)))
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
        swae_logger.info("FID: %f" % self.cur_fid_dist)

        dict_to_save["fid"] = str(self.cur_fid_dist)
        dict_to_save["time_spended"] = str(time.time() - self.start_train_time)
        with open(os.path.join(dir_path, "vars.json"), "w") as f:
            json.dump(dict_to_save, f, indent=4)

        if self.cur_fid_dist <= settings.NEEDED_FID_SCORE:
            swae_logger.info("==== STOP TRAINING ====")
            self.need_break = True

    def _save_generator_model(self, dir_path):
        self.decoder.save(filepath=os.path.join(dir_path, "generator_model.h5"))

    def _save_loss_func(self, dir_path):
        np.save(os.path.join(dir_path, "autoencoder_loss.npy"), self.loss_list)

        fig = plt.figure(figsize=(15, 15))
        plt.plot(np.asarray(self.loss_list))
        fig.savefig(os.path.join(dir_path, "autoencoder_loss.png"))
        plt.close()

    def _save_fid_score_list(self, dir_path):
        np.save(os.path.join(dir_path, "fid_score.npy"), self.fid_score_list)

        fig = plt.figure(figsize=(15, 15))
        plt.plot(np.asarray(self.fid_score_list))
        fig.savefig(os.path.join(dir_path, "fid_score.png"))
        plt.close()

    def _save_checkpoint(self, epoch):
        dir_path = os.path.join(self.checkpoint_dir, "Epoch_%d" % epoch)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self._save_sampled_images(dir_path)
        self._save_current_checkpoint_vars(dir_path)
        self._save_generator_model(dir_path)
        self._save_loss_func(dir_path)
        self._save_fid_score_list(dir_path)

