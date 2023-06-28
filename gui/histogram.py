import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

mask_dir = "./mask/"
image_dir = "./original/"

mask_names = sorted(os.listdir(mask_dir))
image_names = sorted(os.listdir(image_dir))

# method 1
for mask_name, image_name in zip(mask_names, image_names):
    mask = cv2.imread(os.path.join(mask_dir, mask_name))
    img = cv2.imread(os.path.join(image_dir, image_name))
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    indices = np.where(mask[:,:,0]==255)
    hist_h = cv2.calcHist([h[indices]], [0], None, [181], [0, 181])
    hist_s = cv2.calcHist([s[indices]], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v[indices]], [0], None, [256], [0, 256])
    print(np.squeeze(hist_h))
    exit()
    plt.figure(mask_name)
    plt.subplot(121), plt.imshow(img[:,:,::-1]), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.plot(hist_h, color = '#8ECAE6', label = "Hue")
    plt.plot(hist_s, color = '#FB8500', label = "Saturation")
    plt.plot(hist_v, color = '#023047', label = "Value")
    plt.title("HSV histogram")
    plt.legend()
    plt.show()