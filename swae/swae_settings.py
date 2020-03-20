import os


LATENT_DIM = 100
BATCH_SIZE = 64
GP_WEIGHT = 10
IMG_ROWS = 28
IMG_COLS = 28
IMG_CHANNELS = 1
PROJECTIONS_NUM = 50                          # Number of random projections (thetas)

SAMPLE_INTERVAL = 10
CHECKPOINT_INTERVAL = 30

NEEDED_FID_SCORE = 0.43929598

OUTPUT_DIR = os.path.join("result_pictures", "conv_swae")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CHECKPOINT_DIR = os.path.join("saved_models", "conv_swae", "checkpoints")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
