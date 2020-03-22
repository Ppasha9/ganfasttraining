import os
import json
import time
import random

import numpy as np
import tensorflow_probability
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import wae.wae_settings as settings

from tensorflow_gan.examples.mnist import util

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Activation, AveragePooling2D
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from keras import backend as K

from datasets import get_mnist_dataset_rescaled
from logger import wae_logger


_STDDEV = 0.00999

tf.set_random_seed(30)
random.seed(30)


class WAE_MODEL(object):
    def __init__(self):
        self.batch_size = settings.BATCH_SIZE
        self.pics_output_dir = settings.OUTPUT_DIR
        self.checkpoint_dir = settings.CHECKPOINT_DIR
        self.lamda = settings.LAMBDA
        self.latent_dim = settings.LATENT_DIM

        self.encoder_decoder_lr = settings.ENCODER_DECODER_LR
        self.critic_lr = settings.CODE_CRITIC_LR

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

        self.X = tf.placeholder(tf.float32, [None, 28 * 28])
        Y = tf.placeholder(tf.float32, [None, 10])

        def prior_z():
            z_mean = tf.zeros(self.latent_dim)
            z_var = tf.ones(self.latent_dim)
            return tensorflow_probability.distributions.MultivariateNormalDiag(z_mean, z_var)

        prior_dist = prior_z()
        self.z_prime = prior_dist.sample(self.batch_size)

        # z_hat ~ Q(Z|X) variational distribution parametrized by encoder network
        self.z_hat = self._encoder(self.X)
        self.z_hat_test = self._encoder(self.X, is_trainable=False, reuse=True)
        self.x_hat = self._decoder(self.z_hat)
        self.x_hat_test = self._decoder(self.z_hat_test, is_trainable=False, reuse=True)
        self.x_prime = self._decoder(self.z_prime, reuse=True)

        self.z_hat_logits = self._critic(self.z_hat)
        self.z_prime_logits = self._critic(self.z_prime, reuse=True)

        self.l2_recons_loss = tf.reduce_mean(tf.pow(self.X - self.x_hat, 2))

        self.enc_adversary_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z_hat_logits, labels=tf.ones_like(self.z_hat_logits)))

        self.critic_real_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z_prime_logits, labels=tf.ones_like(self.z_prime_logits)))
        self.critic_fake_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z_hat_logits, labels=tf.zeros_like(self.z_hat_logits)))

        self.critic_loss = self.lamda * (self.critic_real_term + self.critic_fake_term)

        self.enc_dec_loss = self.l2_recons_loss + self.lamda * self.enc_adversary_loss

        self.encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="wae_encoder")
        self.decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="wae_decoder")
        self.enc_dec_params = self.encoder_params + self.decoder_params
        self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="wae_critic")

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.enc_dec_optimizer = tf.train.AdamOptimizer(learning_rate=self.encoder_decoder_lr, beta1=0.5)
            self.enc_dec_gradients_vars = self.enc_dec_optimizer.compute_gradients(self.enc_dec_loss, self.enc_dec_params)
            self.enc_dec_train_optimizer = self.enc_dec_optimizer.apply_gradients(self.enc_dec_gradients_vars)

            self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr, beta1=0.5)
            self.critic_gradients_vars = self.critic_optimizer.compute_gradients(self.critic_loss, self.critic_params)
            self.critic_train_optimizer = self.critic_optimizer.apply_gradients(self.critic_gradients_vars)

        tf.summary.scalar("enc_dec_loss", self.enc_dec_loss)
        tf.summary.scalar("critic_loss", self.critic_loss)
        tf.summary.scalar("l2_recons_loss", self.l2_recons_loss)

        self.all_grad_vars = [self.enc_dec_gradients_vars, self.critic_gradients_vars]

        for grad_vars in self.all_grad_vars:
            for g, v in grad_vars:
                tf.summary.histogram(v.name, v)
                tf.summary.histogram(v.name + str('grad'), g)

        self.merged_all = tf.summary.merge_all()

    def _encoder(self, x, reuse=False, is_trainable=True):
        with tf.variable_scope("wae_encoder") as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.reshape(x, [-1, *self.img_shape])

            conv1 = tf.layers.conv2d(x,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=_STDDEV),
                                     filters=16,
                                     kernel_size=[3, 3],
                                     padding="SAME",
                                     strides=(2, 2),
                                     name='enc_conv1_layer',
                                     activation=None,
                                     trainable=is_trainable,
                                     reuse=reuse)
            conv1 = tf.layers.batch_normalization(conv1, training=is_trainable, reuse=reuse, name="bn_1")
            conv1 = tf.nn.relu(conv1, name="leaky_relu_conv1")

            # 14x14x32
            conv2 = tf.layers.conv2d(conv1,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=_STDDEV),
                                     filters=32,
                                     kernel_size=[3, 3],
                                     padding="SAME",
                                     strides=(2, 2),
                                     name='enc_conv2_layer',
                                     activation=None,
                                     trainable=is_trainable,
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2, training=is_trainable, reuse=reuse, name="bn_2")
            conv2 = tf.nn.relu(conv2, name="leaky_relu_conv2")

            # 7x7x64
            conv3 = tf.layers.conv2d(conv2,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=_STDDEV),
                                     filters=64,
                                     kernel_size=[3, 3],
                                     padding="SAME",
                                     strides=(2, 2),
                                     name='enc_conv3_layer',
                                     activation=None,
                                     trainable=is_trainable,
                                     reuse=reuse)
            conv3 = tf.layers.batch_normalization(conv3, training=is_trainable, reuse=reuse, name="bn_3")
            conv3 = tf.nn.relu(conv3, name="leaky_relu_conv3")

            # 4x4x128
            conv3_flattened = tf.layers.flatten(conv3)

            latent_code = tf.layers.dense(conv3_flattened, self.latent_dim,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=_STDDEV),
                                          activation=None,
                                          name="enc_latent_code",
                                          trainable=is_trainable,
                                          reuse=reuse)
            return latent_code

    # def _decoder(self):
    #     model = Sequential()
    #
    #     model.add(Dense(1024, input_dim=self.latent_dim))
    #     model.add(LeakyReLU(alpha=0.2))
    #
    #     model.add(Dense(128 * 7 * 7))
    #     model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    #
    #     model.add(UpSampling2D())
    #     model.add(Conv2D(128, kernel_size=4, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(LeakyReLU(alpha=0.2))
    #
    #     model.add(UpSampling2D())
    #     model.add(Conv2D(64, kernel_size=4, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(LeakyReLU(alpha=0.2))
    #
    #     model.add(Conv2D(self.img_channels, kernel_size=4, padding="same"))
    #     model.add(Activation("sigmoid"))
    #
    #     model.summary()
    #
    #     noise_input = Input(shape=(self.latent_dim,))
    #     generated_img = model(noise_input)
    #
    #     return Model(noise_input, generated_img)

    def _decoder(self, x, is_trainable=True, reuse=False):
        with tf.variable_scope("wae_decoder") as scope:
            if reuse:
                scope.reuse_variables()

            fc1 = tf.layers.dense(x, 1024, activation=None, name='decoder_fc1', trainable=is_trainable, reuse=reuse)
            fc1 = tf.nn.relu(fc1)

            fc2 = tf.layers.dense(fc1, 128 * 7 * 7, activation=None, name='decoder_fc2', trainable=is_trainable, reuse=reuse)
            fc2 = tf.keras.layers.Reshape((7, 7, 128), input_shape=(128 * 7 * 7,))(fc2)

            conv1 = tf.keras.layers.UpSampling2D()(fc2)
            conv1 = tf.layers.conv2d(conv1,
                                     filters=128,
                                     kernel_size=[4, 4],
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=_STDDEV),
                                     padding="SAME",
                                     name='decoder_conv1',
                                     trainable=is_trainable,
                                     reuse=reuse)
            conv1 = tf.layers.batch_normalization(conv1, momentum=0.8, trainable=is_trainable, reuse=reuse, name='decoder_bn1')
            conv1 = tf.nn.relu(conv1)

            conv2 = tf.keras.layers.UpSampling2D()(conv1)
            conv2 = tf.layers.conv2d(conv2,
                                     filters=64,
                                     kernel_size=[4, 4],
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=_STDDEV),
                                     padding="SAME",
                                     name='decoder_conv2',
                                     trainable=is_trainable,
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2, momentum=0.8, trainable=is_trainable, reuse=reuse, name='decoder_bn2')
            conv2 = tf.nn.relu(conv2)

            conv3 = tf.layers.conv2d(conv2,
                                     filters=self.img_channels,
                                     kernel_size=[4, 4],
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=_STDDEV),
                                     padding="SAME",
                                     name='decoder_conv2',
                                     trainable=is_trainable,
                                     reuse=reuse)
            decoded = tf.nn.sigmoid(conv3)

            return decoded

    def _critic(self, z, is_trainable=True, reuse=False):
        with tf.variable_scope("wae_critic") as scope:
            if reuse:
                scope.reuse_variables()

            fc1 = tf.layers.dense(z, 512, activation=None, name='code_critic_fc1', trainable=is_trainable, reuse=reuse)
            fc1 = tf.nn.relu(fc1)

            fc2 = tf.layers.dense(fc1, 512, activation=None, name='code_critic_fc2', trainable=is_trainable, reuse=reuse)
            fc2 = tf.nn.relu(fc2)

            fc3 = tf.layers.dense(fc2, 512, activation=None, name='code_critic_fc3', trainable=is_trainable, reuse=reuse)
            fc3 = tf.nn.relu(fc3)

            logits = tf.layers.dense(fc3, 1, activation=None, name='code_critic_fc4', trainable=is_trainable, reuse=reuse)
            return logits

    def train(self):
        # load dataset
        x_train, _, x_test, _ = get_mnist_dataset_rescaled()

        self.x_train = x_train
        self.x_fid_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

        n_batches = int(self.x_train.shape[0] / self.batch_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('-' * 80)
            print('n_batches : ', n_batches, ' when batch_size : ', self.batch_size)
            # for tensorboard
            saver = tf.train.Saver(max_to_keep=3)
            writer = tf.summary.FileWriter(settings.LOG_DIRECTORY, sess.graph)
            iterations = 0

            epoch = 0
            self.start_train_time = time.time()
            while not self.need_break:
                for batch in range(n_batches):
                    iterations += 1

                    ind = np.random.permutation(self.x_train.shape[0])

                    # Train Code Discriminator
                    for i in range(1):
                        next_batch = x_train[ind[i * self.batch_size: (i + 1) * self.batch_size], ...]
                        fd = {self.X: next_batch}
                        _, critic_loss = sess.run([self.critic_train_optimizer, self.critic_loss], feed_dict=fd)

                    # Train Encoder
                    for i in range(1):
                        next_batch = x_train[ind[i * self.batch_size: (i + 1) * self.batch_size], ...]
                        fd = {self.X: next_batch}
                        _, _enc_dec_loss, merged = sess.run([self.enc_dec_train_optimizer, self.enc_dec_loss, self.merged_all], feed_dict=fd)

                    if iterations % 20 == 0:
                        writer.add_summary(merged, iterations)

                    if batch % 200 == 0:
                        wae_logger('Batch #', batch, ' done!')

                if epoch % 5 == 0:
                    n = 5

                    reconstructed = np.empty((28 * n, 28 * n))
                    original = np.empty((28 * n, 28 * n))
                    generated = np.empty((28 * n, 28 * n))

                    ind = np.random.permutation(x_test.shape[0])
                    for i in range(n):
                        next_batch = x_test[ind[i * self.batch_size: (i + 1) * self.batch_size], ...]
                        recons = sess.run(self.x_hat_test, feed_dict={self.X: next_batch})
                        recons = np.reshape(recons, [-1, 784])

                        sample = tf.random_normal([n, self.latent_dim])
                        generation = sess.run(self.x_hat_test, feed_dict={self.z_hat_test: sample.eval()})

                        for j in range(n):
                            original[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = next_batch[j].reshape([28, 28])

                        for j in range(n):
                            reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28])

                        for j in range(n):
                            generated[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = generation[j].reshape([28, 28])

                    print("Generated Images")
                    plt.figure(figsize=(n, n))
                    plt.axis('off')
                    plt.imshow(generated, origin="upper", cmap="gray")
                    plt.title('Epoch ' + str(epoch))
                    plt.savefig(os.path.join(settings.OUTPUT_DIR, 'Epoch_' + str(epoch), 'gen-img.png'))
                    plt.close()

                    print("Original Images")
                    plt.figure(figsize=(n, n))
                    plt.axis('off')
                    plt.imshow(original, origin="upper", cmap="gray")
                    plt.title('Epoch ' + str(epoch))
                    plt.savefig(os.path.join(settings.OUTPUT_DIR, 'Epoch_' + str(epoch), 'orig-img.png'))
                    plt.close()

                    print("Reconstructed Images")
                    plt.figure(figsize=(n, n))
                    plt.axis('off')
                    plt.imshow(reconstructed, origin="upper", cmap="gray")
                    plt.title('Epoch ' + str(epoch))
                    plt.savefig(os.path.join(settings.OUTPUT_DIR, 'Epoch_' + str(epoch), 'recons-img.png'))
                    plt.close()

                if epoch % 5 == 0:
                    save_path = saver.save(sess, os.path.join(settings.CHECKPOINT_DIR, 'Epoch_' + str(epoch)))
                    print("At epoch #", epoch, " Model is saved at path: ", save_path)

                print('------------------------------------')
                print('=== Epoch #', epoch, ' completed! ===')
                print('------------------------------------')
