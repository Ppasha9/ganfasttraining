from keras.layers import Flatten, Input, Dense, Reshape
from keras.models import Model

from ._autoencoder import AutoEncoder


class DenseAutoEncoder(AutoEncoder):
    ENCODER_MODEL_JSON = "D:\\University\\Diploma\\VAE\\saved_models\\dense_ae\\encoder_model.json"
    ENCODER_WEIGHTS = "D:\\University\\Diploma\\VAE\\saved_models\\dense_ae\\encoder_weights.h5"
    DECODER_MODEL_JSON = "D:\\University\\Diploma\\VAE\\saved_models\\dense_ae\\decoder_model.json"
    DECODER_WEIGHTS = "D:\\University\\Diploma\\VAE\\saved_models\\dense_ae\\decoder_weights.h5"
    AUTOENCODER_MODEL_JSON = "D:\\University\\Diploma\\VAE\\saved_models\\dense_ae\\autoencoder_model.json"
    AUTOENCODER_WEIGHTS = "D:\\University\\Diploma\\VAE\\saved_models\\dense_ae\\autoencoder_weights.h5"

    def _create_models(self):
        # Encoder
        # Input placeholder
        input_img = Input(shape=(28, 28, 1))
        # Reshaping layer
        flat_img = Flatten()(input_img)
        encoded = Dense(self._latent_dim, activation='relu')(flat_img)

        # Decoder
        input_encoded = Input(shape=(self._latent_dim,))
        flat_decoded = Dense(28 * 28, activation='sigmoid')(input_encoded)
        decoded = Reshape((28, 28, 1))(flat_decoded)

        self._encoder_model = Model(input_img, encoded, name='encoder')
        self._decoder_model = Model(input_encoded, decoded, name='decoder')
        self._ae_model = Model(input_img, self._decoder_model(self._encoder_model(input_img)), name='Dense AutoEncoder')

    def _get_optimizer(self):
        return 'adam'

    def _get_loss_func(self):
        return 'binary_crossentropy'
