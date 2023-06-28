from network.unet_vgg import VGG, Unet_VGG
from data import ValidationDataset
from network.unet import Unet
# from torchsummary import summary
import torch
import torchvision
from torch.utils.data import DataLoader
from data import HairSegmentationDataset
from data import AugumentationDataset
from config import *
from network.unet import Unet 
from torchvision import transforms

import time
import tqdm
import numpy as np
from metrics import *
from evaluate import *
import os
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
from torchsummary import summary
import torch.nn as nn

vgg = VGG()
temp = torchvision.models.vgg16(pretrained=True)
vgg.load_state_dict(temp.state_dict(), strict = False)
# freeze layers
for param in vgg.parameters():
    param.requires_grad = False
model = Unet_VGG(vgg)
# input = torch.rand(1, 3, 512, 512)
# vgg16 = models.vgg16(pretrained=True)
# output = model(input)
# print(output)
# summary(model, (3, 512, 512))
# exit()
# summary(my_model, (3, 512, 512))

loader_params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 0}

loader_params_test = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0}

transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

train_data = HairSegmentationDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, RESIZE_SIZE, transforms)
aug_data = AugumentationDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, RESIZE_SIZE, transforms)
test_data = ValidationDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, transforms)

train_data = torch.utils.data.ConcatDataset([train_data, aug_data])

train_set = torch.utils.data.DataLoader(train_data, **loader_params)
test_set = torch.utils.data.DataLoader(test_data, **loader_params_test)

# Select gpu if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize model
model.to(device)



loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#initialize tensorboard
# writer = SummaryWriter("./runs/" + LOG_DIR_NAME)

# logging.basicConfig(filename = "./logs/" + LOG_DIR_NAME + ".log", level = logging.INFO)
# log = logging.getLogger()

# log.info(f"Date: {datetime.now()}")
# log.info(f"Learning rate: {LR}")
# log.info(f"Batch size: {BATCH_SIZE}")
# log.info(f"Resize size (for training): {RESIZE_SIZE}")
# log.info(f"Epochs: {EPOCHS}")
# log.info(f"Save directory: {MODEL_SAVE_DIR}")

print("========> Starting training")
start_time = time.time()
offset = 250
for epoch in tqdm.tqdm(range(EPOCHS)):
    model.train()
    epoch = epoch
    tr_loss = 0
    train_loss = 0
    val_loss = 0
    train_step = 0
    val_step = 0
    iou_score = 0
    accuracy = 0
    precision = 0
    f1_score = 0
    train_acc = 0

    for(i,(image,mask)) in enumerate(tqdm.tqdm(train_set)):
        (image, mask) = (image.to(device), mask.to(device))

        features = model(image)
        print(features.shape)
        exit()
        train_loss = loss_function(prediction, mask)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        tr_loss += train_loss
        train_acc += train_accuracy(prediction, mask)
        train_step += 1
