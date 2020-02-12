import numpy
import seaborn
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam


if __name__ == "__main__":
    # Dataset build
    x1 = numpy.linspace(-2.2, 2.2, num=1000)
    fx = numpy.sin(x1)
    dots1 = numpy.vstack([x1, fx]).T

    t = numpy.linspace(0, 2 * numpy.pi, num=1000)
    dots2 = 0.5 * numpy.array([numpy.sin(t), numpy.cos(t)]).T + numpy.array([1.5, -0.5])[None, :]

    dots = numpy.vstack([dots1, dots2])
    noise = 0.06 * numpy.random.randn(*dots.shape)

    labels = numpy.array([0] * 1000 + [1] * 1000)
    noised = dots + noise

    # Model training
    dae = Sequential()
    dae.add(Dense(64, activation='elu', input_dim=2))
    dae.add(Dense(64, activation='elu'))
    dae.add(Dense(1, activation='linear'))
    dae.add(Dense(64, activation='elu'))
    dae.add(Dense(64, activation='elu'))
    dae.add(Dense(2, activation='linear'))

    input_dots = Input(shape=(2,))
    dae_model = Model(input_dots, dae(input_dots), name="Simple Deep AutoEncoder")

    dae_model.compile(optimizer=Adam(0.001), loss="mse")
    dae_model.summary()
    dae_model.fit(noised, noised, epochs=300, batch_size=30, verbose=2)

    # Result
    predicted = dae_model.predict(noised)

    # Visualization
    colors = ['b'] * 1000 + ['g'] * 1000
    plt.figure(figsize=(15, 9))
    plt.xlim([-2.5, 2.5])
    plt.ylim([-1.5, 1.5])
    plt.scatter(noised[:, 0], noised[:, 1], c=colors)
    plt.plot(dots1[:, 0], dots1[:, 1], color="red", linewidth=4)
    plt.plot(dots2[:, 0], dots2[:, 1], color="yellow", linewidth=4)
    plt.scatter(predicted[:, 0], predicted[:, 1], color="black")
    plt.show()

