from keras.layers import Dense, Input
from keras.layers import BatchNormalization, Dropout, Flatten, Reshape, Lambda
from keras.models import Model

from keras.optimizers import Adam
from keras.objectives import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

from ._autoencoder import AutoEncoder


class VariationalAutoEncoder(AutoEncoder):
    ENCODER_MODEL_JSON = "D:\\University\\Diploma\\VAE\\saved_models\\variational_ae\\encoder_model.json"
    ENCODER_WEIGHTS = "D:\\University\\Diploma\\VAE\\saved_models\\variational_ae\\encoder_weights.h5"
    DECODER_MODEL_JSON = "D:\\University\\Diploma\\VAE\\saved_models\\variational_ae\\decoder_model.json"
    DECODER_WEIGHTS = "D:\\University\\Diploma\\VAE\\saved_models\\variational_ae\\decoder_weights.h5"
    AUTOENCODER_MODEL_JSON = "D:\\University\\Diploma\\VAE\\saved_models\\variational_ae\\autoencoder_model.json"
    AUTOENCODER_WEIGHTS = "D:\\University\\Diploma\\VAE\\saved_models\\variational_ae\\autoencoder_weights.h5"

    def _create_models(self):
        def apply_bn_and_dropout(x):
            return Dropout(self._dropout_rate)(BatchNormalization()(x))

        input_img = Input(batch_shape=(self._batch_size, 28, 28, 1))
        x = Flatten()(input_img)
        x = Dense(256, activation='relu')(x)
        x = apply_bn_and_dropout(x)
        x = Dense(128, activation='relu')(x)
        x = apply_bn_and_dropout(x)

        z_mean = Dense(self._latent_dim)(x)
        z_log_var = Dense(self._latent_dim)(x)

        # for  valid serializing the encoder's model
        batch_size = self._batch_size
        latent_dim = self._latent_dim

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        self._encoder_model = Model(input_img, l, name="encoder")
        self._z_meaner = Model(input_img, z_mean, name="encoder_z_meaner")
        self._z_log_varer = Model(input_img, z_log_var, name="encoder_z_varer")

        z = Input(shape=(self._latent_dim,))
        x = Dense(128)(z)
        x = LeakyReLU()(x)
        x = apply_bn_and_dropout(x)
        x = Dense(256)(x)
        x = LeakyReLU()(x)
        x = apply_bn_and_dropout(x)
        x = Dense(28 * 28, activation='sigmoid')(x)
        decoded = Reshape((28, 28, 1))(x)

        self._decoder_model = Model(z, decoded, name="decoder")
        self._ae_model = Model(input_img, self._decoder_model(self._encoder_model(input_img)), name="variational_autoencoder")

        def vae_loss(x, decoded):
            x = K.reshape(x, shape=(self._batch_size, 28 * 28))
            decoded = K.reshape(decoded, shape=(self._batch_size, 28 * 28))
            xent_loss = 28 * 28 * binary_crossentropy(x, decoded)
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return (xent_loss + kl_loss) / 2 / 28 / 28

        self.vae_loss_func = vae_loss

    def _get_optimizer(self):
        return Adam(self._start_learning_rate)

    def _get_loss_func(self):
        return self.vae_loss_func
