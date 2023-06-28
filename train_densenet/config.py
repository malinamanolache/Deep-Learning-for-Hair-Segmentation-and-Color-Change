# Paths to dataset
TRAIN_IMAGE_DIR = '/shared_storage/manolac/dataset/train/data/hair/'
TRAIN_MASK_DIR = '/shared_storage/manolac/dataset/train/mask/'

TEST_IMAGE_DIR = '/shared_storage/manolac/dataset/test/data/hair/'
TEST_MASK_DIR = '/shared_storage/manolac/dataset/test/mask/'

# Training parameters
LR = 0.0001
EPOCHS = 250
LOG_DIR_NAME = "/shared_storage/manolac/densenet/logs/densnet1"
RESIZE_SIZE = 512
BATCH_SIZE = 2
MODEL_SAVE_DIR = "/shared_storage/manolac/densenet/models/"
MODEL_PATH_TRAIN = ""
TENSORBOARD_PATH = "/shared_storage/manolac/densenet/runs/densenet1"

# Inference parameters
MODEL_PATH = ""
INPUT_DIR = ""
OUTPUT_DIR = ""
