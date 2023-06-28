import numpy as np
import torch
from metrics import *

def prepare_for_evaluation(prediction, gt, for_train=False):

    prediction = prediction.cpu().detach().numpy()
    prediction = np.squeeze(prediction, axis = 0)
    prediction = (prediction > 0.5).astype(np.uint8)
    
    gt = gt.cpu().detach().numpy()
    gt = np.squeeze(gt, axis = 0)
    gt = gt.astype(np.uint8)

    return prediction, gt

def evaluate(prediction, gt):
    accuracy = pixel_accuracy(prediction, gt)
    iou_score = iou(prediction, gt)
    precision = pixel_precision(prediction, gt)
    f1_score = f1(prediction, gt)

    return accuracy, iou_score, precision

def train_accuracy(prediction, gt):
    prediction = prediction.cpu().detach().numpy()
    prediction = (prediction > 0.5).astype(np.uint8)

    gt = gt.cpu().detach().numpy()
    gt = gt.astype(np.uint8)

    accuracy = pixel_accuracy(prediction, gt)

    return accuracy
    