import os
import abc

from keras.models import model_from_json


class AutoEncoder(abc.ABC):
    def __init__(self, latent_dim, batch_size, dropout_rate=None, start_learning_rate=None):
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._dropout_rate = dropout_rate
        self._start_learning_rate = start_learning_rate

        self.loaded = False

        if os.path.exists(self.ENCODER_MODEL_JSON):
            self.deserialize()
        else:
            self._create_models()

    def compile(self):
        self._ae_model.compile(optimizer=self._get_optimizer(), loss=self._get_loss_func())
        self._ae_model.summary()

    def fit(self, x, y, epochs_num, validation_data):
        self._ae_model.fit(x, y,
                           epochs=epochs_num,
                           shuffle=True,
                           batch_size=self._batch_size,
                           validation_data=validation_data)

    def predict(self, imgs, batch_size):
        return self._ae_model.predict(imgs, batch_size=batch_size)

    def get_decoder_model(self):
        return self._decoder_model

    def serialize(self):
        if os.path.exists(self.ENCODER_MODEL_JSON):
            return

        with open(self.ENCODER_MODEL_JSON, "w") as f:
            f.write(self._encoder_model.to_json())
        self._encoder_model.save_weights(self.ENCODER_WEIGHTS)

        with open(self.DECODER_MODEL_JSON, "w") as f:
            f.write(self._decoder_model.to_json())
        self._decoder_model.save_weights(self.DECODER_WEIGHTS)

        with open(self.AUTOENCODER_MODEL_JSON, "w") as f:
            f.write(self._ae_model.to_json())
        self._ae_model.save_weights(self.AUTOENCODER_WEIGHTS)

    def deserialize(self):
        if not os.path.exists(self.ENCODER_MODEL_JSON):
            return

        with open(self.ENCODER_MODEL_JSON, "r") as f:
            data = f.read()
        self._encoder_model = model_from_json(data)
        self._encoder_model.load_weights(self.ENCODER_WEIGHTS)

        with open(self.DECODER_MODEL_JSON, "r") as f:
            data = f.read()
        self._decoder_model = model_from_json(data)
        self._decoder_model.load_weights(self.DECODER_WEIGHTS)

        with open(self.AUTOENCODER_MODEL_JSON, "r") as f:
            data = f.read()
        self._ae_model = model_from_json(data)
        self._ae_model.load_weights(self.AUTOENCODER_WEIGHTS)

        self.loaded = True

    @abc.abstractmethod
    def _create_models(self):
        pass

    @abc.abstractmethod
    def _get_optimizer(self):
        pass

    @abc.abstractmethod
    def _get_loss_func(self):
        pass
