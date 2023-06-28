import numpy as np
import torch

def iou(mask1, mask2):

    intersection = np.sum(np.logical_and(mask1, mask2)) # returns 1 when both are 1
    union = np.sum(np.logical_or(mask1, mask2))
    iou_score = intersection / union

    return iou_score


def pixel_accuracy(prediction, gt):

    acc = np.sum(prediction == gt)/(gt.size)

    return acc


def pixel_precision(prediction, gt):
    true_positive = np.sum(np.logical_and(gt, prediction))
    all_positive = np.sum(prediction == 1)
    precision = true_positive/all_positive
    if np.isnan(precision):
        precision=0

    return precision


def f1(prediction, gt):

    intersection = np.logical_and(prediction, gt) # returns 1 when both are 1
    f1_score = 2 * np.sum(intersection) / (2*(gt.size))

    return f1_score
