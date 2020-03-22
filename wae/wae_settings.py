import os

ENCODER_DECODER_LR = 0.001
CODE_CRITIC_LR = 0.0005
BATCH_SIZE = 64
LATENT_DIM = 100
LAMBDA = 1
IMG_ROWS = 28
IMG_COLS = 28
IMG_CHANNELS = 1

SAMPLE_INTERVAL = 2
CHECKPOINT_INTERVAL = 4

NEEDED_FID_SCORE = 0.2604017

OUTPUT_DIR = os.path.join("result_pictures", "conv_swae")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CHECKPOINT_DIR = os.path.join("saved_models", "conv_swae", "checkpoints")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

LOG_DIRECTORY = "./log-wae-train"
if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)
