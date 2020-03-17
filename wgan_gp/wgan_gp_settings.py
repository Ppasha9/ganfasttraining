import os


LATENT_DIM = 100
BATCH_SIZE = 64
CRITIC_UPDATES_ITERS = 5  # number of critic updates per generator update
GP_WEIGHT = 10
IMG_ROWS = 28
IMG_COLS = 28
IMG_CHANNELS = 1

SAMPLE_INTERVAL = 10
CHECKPOINT_INTERVAL = 50

OUTPUT_DIR = os.path.join("result_pictures", "conv_wgan_gp")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CHECKPOINT_DIR = os.path.join("saved_models", "conv_wgan_gp", "checkpoints")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
