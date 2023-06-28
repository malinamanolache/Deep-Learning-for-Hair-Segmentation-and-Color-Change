from data import ValidationDataset
# from torchsummary import summary
import torch
import torchvision
from torch.utils.data import DataLoader
from data import HairSegmentationDataset
from data import AugumentationDataset
from config import *
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
import torchextractor as tx

from densenet import Densenet_Unet


# Prepare data
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

model = Densenet_Unet()
model.to(device)

if MODEL_PATH_TRAIN != "":
    print("Restoring model from: ", MODEL_PATH_TRAIN)
    model.load_state_dict(torch.load(MODEL_PATH_TRAIN))
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#initialize tensorboard
writer = SummaryWriter(TENSORBOARD_PATH)

logging.basicConfig(filename = LOG_DIR_NAME + ".log", level = logging.INFO)
log = logging.getLogger()

log.info(f"Date: {datetime.now()}")
log.info(f"Learning rate: {LR}")
log.info(f"Batch size: {BATCH_SIZE}")
log.info(f"Resize size (for training): {RESIZE_SIZE}")
log.info(f"Epochs: {EPOCHS}")
log.info(f"Save directory: {MODEL_SAVE_DIR}")

print("========> Starting training")
start_time = time.time()
offset = 0
for epoch in tqdm.tqdm(range(EPOCHS)):
    model.train()
    epoch = epoch+offset
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

        prediction = model(image)
        train_loss = loss_function(prediction, mask)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        tr_loss += train_loss
        train_acc += train_accuracy(prediction, mask)
        train_step += 1

    with torch.no_grad():
        model.eval()

        # loop over the validation set
        for (val_image, val_mask) in test_set:
            # send the input to the device

            (val_image, val_mask) = (val_image.to(device), val_mask.to(device))
            # make the predictions and calculate the validation loss
            val_prediction = model(val_image)
            val_loss += loss_function(val_prediction, val_mask)

            val_prediction_ev, val_gt_ev = prepare_for_evaluation(val_prediction, val_mask)
            accuracy += pixel_accuracy(val_prediction_ev, val_gt_ev)
            precision += pixel_precision(val_prediction_ev, val_gt_ev)
            iou_score += iou(val_prediction_ev, val_gt_ev)

            val_step += 1

    tr_loss  = tr_loss.cpu().detach().numpy()
    val_loss  = val_loss.cpu().detach().numpy()
    train_loss = tr_loss/train_step
    writer.add_scalar("Train loss", train_loss, epoch)
    writer.add_scalar("Train accuracy", train_acc/train_step, epoch)
    writer.add_scalar("Validation loss", val_loss/val_step, epoch)
    writer.add_scalar("Accuracy", accuracy/val_step, epoch)
    writer.add_scalar("IoU", iou_score/val_step, epoch)
    writer.add_scalar("Precision", precision/val_step, epoch)

    log.info(f"Epoch:{epoch} - Train loss:{tr_loss} - Train accuracy:{train_acc/train_step} - Val loss:{val_loss/val_step} - Accuracy:{accuracy/val_step}% - IoU:{iou_score/val_step}% \
             - Precison:{precision/val_step}%")
    # save model
    torch.save(model.state_dict(), MODEL_SAVE_DIR + str(epoch) + ".pth")


end_time = time.time()
training_time = end_time - start_time
log.info("Training finished")
log.info("Training time: ", training_time)
writer.close()