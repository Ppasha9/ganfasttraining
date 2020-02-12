import os
import matplotlib.pyplot as plt

from keras.utils.vis_utils import plot_model

# from autoencoder import DenseAutoEncoder
from autoencoder import VariationalAutoEncoder, DenseAutoEncoder
from datasets import get_mnist_dataset


def plot_digits(first_row, second_row):
    plt.figure(figsize=(20, 4))
    n = len(first_row)
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(first_row[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(second_row[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_mnist_dataset()

    # autoencoder = DenseAutoEncoder(latent_dim=49, batch_size=256, dropout_rate=0.3)

    # if not autoencoder.loaded:
    #     autoencoder.compile()
    #     autoencoder.fit(x=x_test, y=x_test, epochs_num=50, validation_data=(x_test, x_test))

    # os.environ["PATH"] += os.pathsep + 'C:/Users/User/Downloads/graphviz-2.38/release/bin/'
    # x = DenseAutoEncoder(latent_dim=100, batch_size=500, dropout_rate=0.3, start_learning_rate=0.0001)
    # x = VariationalAutoEncoder(latent_dim=100, batch_size=500, dropout_rate=0.3, start_learning_rate=0.0001)
    # plot_model(x._encoder_model, to_file="saved_models\\conditional_variational_ae\\test_encoder_vae.png", rankdir='LR', show_layer_names=False)
    # plot_model(x._decoder_model, to_file="saved_models\\conditional_variational_ae\\test_decoder_vae.png", rankdir='LR', show_layer_names=False)
    #

    autoencoder = VariationalAutoEncoder(latent_dim=100, batch_size=500, dropout_rate=0.3, start_learning_rate=0.0001)

    if not autoencoder.loaded:
        autoencoder.compile()
        autoencoder.fit(x=x_test, y=x_test, epochs_num=100, validation_data=(x_test, x_test))
        # autoencoder.serialize()

    n = 10
    decoded_imgs = autoencoder.predict(x_test, batch_size=500)
    plot_digits(first_row=x_test[:n], second_row=decoded_imgs[:n])
