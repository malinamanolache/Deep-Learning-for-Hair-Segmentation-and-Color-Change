from core.metrics import *
import cv2
from matplotlib import pyplot as plt

original_path = "/media/intern2/Data/licenta/figaro1k/Figaro1k/GT/testare metrici/originala.png"
test_image_path = "/media/intern2/Data/licenta/figaro1k/Figaro1k/GT/testare metrici/modificata2.png"

original = cv2.imread(original_path)
test_image = cv2.imread(test_image_path)

# plt.figure(1)
# plt.imshow(original)
# plt.show()

# print(original)
# print(test_image)

original = original/255
test_image = test_image/255

acc_modified = pixel_accuracy(test_image, original)
iou_score_modified = iou(test_image, original)
precision_modified = pixel_precision(test_image, original)
f1_score_modified = f1(test_image, original)

acc_original = pixel_accuracy(original, original)
iou_score_original = iou(original, original)
precision_original = pixel_precision(original, original)
f1_score_original = f1(original, original)


print(f"Same image: Accuracy = {acc_original}, iou = {iou_score_original}, precision = {precision_original}, f1 = {f1_score_original}")
print(f"Modified image: Accuracy = {acc_modified}, iou = {iou_score_modified}, precision = {precision_modified}, f1 = {f1_score_modified}")