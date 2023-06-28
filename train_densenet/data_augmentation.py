import numpy as np
import random
import cv2

def rotate(image, mask, angle_range = (7, 18)):

    height, width, _ = image.shape

    cX, cY = width // 2, height // 2
    angle = random.randint(angle_range[0], angle_range[1])
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.4)

    rotated_image = cv2.warpAffine(image, M, (width, height))
    rotated_mask = cv2.warpAffine(mask, M, (width, height))

    return rotated_image, rotated_mask


def random_crop(image, mask, crop_proportions = (0.7, 0.9)):

    height, width, _ = image.shape

    new_width = int(width * random.uniform(crop_proportions[0], crop_proportions[1]))
    new_height = int(height * random.uniform(crop_proportions[0], crop_proportions[1]))

    cX = random.randint(new_width//2, width - new_width//2)
    cY = random.randint(new_height//2, height - new_height//2)

    cropped_image = image[cX-new_width//2:cX+new_width//2, cY-new_height//2:cY+new_height//2, :]
    cropped_mask = mask[cX-new_width//2:cX+new_width//2, cY-new_height//2:cY+new_height//2, :]
    
    return cropped_image, cropped_mask