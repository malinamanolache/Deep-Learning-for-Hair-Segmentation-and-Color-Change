from core.config import *
from core.data import *
import torch
from torch.utils.data import DataLoader

loader_params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 0}
transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
train_data = HairSegmentationDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, transforms)
test_data = HairSegmentationDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, transforms)
dataset = torch.utils.data.ConcatDataset([train_data, test_data])
dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in dataloader:
        # print(data.shape)
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

mean, std = get_mean_and_std(dataloader)
print("Mean of dataset: ", mean)
print("Std of dataset: ", std)

# Mean of dataset:  tensor([0.4302, 0.4783, 0.5437])
# Std of dataset:  tensor([0.2992, 0.2968, 0.2975])