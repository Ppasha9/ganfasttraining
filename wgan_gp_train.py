from wgan_gp import WGAN_GP_MODEL_FOR_MNIST


if __name__ == "__main__":
    wgan_gp = WGAN_GP_MODEL_FOR_MNIST()
    wgan_gp.train()
