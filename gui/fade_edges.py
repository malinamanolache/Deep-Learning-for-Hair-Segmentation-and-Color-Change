import cv2
import os
import numpy as np
import tqdm
from matplotlib import pyplot as plt

mask_dir = "./mask/"
image_dir = "./original/"
save_dir = "./blur_3x3/"
blended_dir = "./blended/"
not_blended_dir = "./not_blended/"
save_blur_dir = "./gaussian_blur_3x3/"

mask_names = sorted(os.listdir(save_dir))
image_names = sorted(os.listdir(image_dir))

def change_color(img, mask):
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    indices_h = np.where(mask[:,:,0]!=0)
    indices_v = np.where(np.logical_and(mask[:,:,0]<255, mask[:,:,0]>0))
    h[indices_h] = 100
    new_img = cv2.merge([h, s, v])
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    
    new_img = new_img.astype(float)
    img = img.astype(float)
    mask = mask.astype(float)/255
    blended = mask*new_img + (1-mask)*img
    blended = blended.astype("uint8")
    new_img = new_img.astype("uint8")
    
    return new_img, blended

for mask_name, image_name in tqdm.tqdm(zip(mask_names, image_names)):
    mask = cv2.imread(save_dir+mask_name)
    image = cv2.imread(image_dir+image_name)
    # new_image, blended = change_color(image, mask)
   
    blurred_mask = cv2.GaussianBlur(mask, (3, 3), 0)
    cv2.imwrite(save_blur_dir+mask_name, blurred_mask)
    # cv2.imwrite(not_blended_dir+mask_name, new_image)
