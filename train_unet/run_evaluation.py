import cv2
import os
from core.metrics import *
import tqdm

gt_dir = "/shared_storage/manolac/dataset/test/mask/"
pred_dir = "/shared_storage/manolac/PyTorch_YOLOv4/yolov4_preds_best/final_masks_pretrained_vgg_177/"
print(pred_dir)

accuracy = 0
precision = 0
iou_score = 0

gt_masks = sorted(os.listdir(gt_dir))
predictions = sorted(os.listdir(pred_dir))

precision_list = []
for prediction in tqdm.tqdm(predictions):
    frame_name = prediction.split("-")[0]
    gt_name = frame_name + "-gt" + ".png"
    pred = cv2.imread(pred_dir + prediction)
    gt = cv2.imread(gt_dir + gt_name)
    # print(gt_dir + gt_name)
    pred = pred/255
    gt = gt/255
    accuracy += pixel_accuracy(pred, gt)
    precision += pixel_precision(pred, gt)
    precision_list.append(pixel_precision(pred, gt))
    iou_score += iou(pred, gt)


accuracy = accuracy / len(gt_masks) * 100
precision = precision / len(gt_masks) * 100
iou_score = iou_score / len(gt_masks) * 100

print(f"Metrics for the {len(gt_masks)} samples:")
print(f"Accuracy = {accuracy}%")
print(f"Precision = {precision}%")
print(f"IoU = {iou_score}%")