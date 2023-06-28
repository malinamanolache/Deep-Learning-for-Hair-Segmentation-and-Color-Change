# Paths to dataset
TRAIN_IMAGE_DIR = '/shared_storage/manolac/dataset/train/data/hair/'
TRAIN_MASK_DIR = '/shared_storage/manolac/dataset/train/mask/'

TEST_IMAGE_DIR = '/shared_storage/manolac/dataset/test/data/hair/'
TEST_MASK_DIR = '/shared_storage/manolac/dataset/test/mask/'

# Training parameters
LR = 0.0001
EPOCHS = 250
LOG_DIR_NAME = "/shared_storage/manolac/network_vgg/unet_vgg/core/logs/unet_cropped_pretrained_vgg"
RESIZE_SIZE = 512
BATCH_SIZE = 6
MODEL_SAVE_DIR = "/shared_storage/manolac/network_vgg/unet_vgg/core/models_cropped_pretrained_vgg"
MODEL_PATH_TRAIN = ""
TENSORBOARD_PATH = "/shared_storage/manolac/network_vgg/unet_vgg/core/runs/models_cropped_pretrained_vgg"

# Inference parameters
MODEL_PATH = "/shared_storage/manolac/final_models_licenta/U-Net_pretrained_109.pth"
INPUT_DIR = "/shared_storage/manolac/dataset/test/data/hair/"
OUTPUT_DIR = "/shared_storage/manolac/final_preds/unet_pretrained/"