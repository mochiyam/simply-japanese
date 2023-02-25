MODEL_NAME = "sonoisa/t5-base-japanese"
MAX_TOKEN_INPUT_LENGTH = 1024
MAX_TOKEN_TARGET_LENGTH = 128

# Column names
INPUT_COL_NAME = 'original'
LABEL_COL_NAME = 'simplified'

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 50
