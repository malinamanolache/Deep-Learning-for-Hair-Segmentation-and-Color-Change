import os
import numpy as np
import cv2
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
    h[indices] = 100
    image_h = cv2.merge([h, s, v])
    image_h = cv2.cvtColor(image_h, cv2.COLOR_HSV2RGB)

    v = np.array(v, dtype=np.float32)
    v[indices] = v[indices]+20
    v = np.clip(v, 0, 255)
    v = v.astype(np.uint8)
    image_v = cv2.merge([h, s, v])
    image_v = cv2.cvtColor(image_v, cv2.COLOR_HSV2RGB)

    s = np.array(s, dtype=np.float32)
    s[indices] = s[indices]+150
    s = np.clip(s, 0, 255)
    s = s.astype(np.uint8)
    image_s = cv2.merge([h, s, v])
    image_s = cv2.cvtColor(image_s, cv2.COLOR_HSV2RGB)

    
#     plt.figure()
#     plt.subplot(151), plt.imshow(img[:,:,::-1]), plt.title("Original image"), plt.xticks([]), plt.yticks([])
#     plt.subplot(152), plt.imshow(mask), plt.title("Binary mask"), plt.xticks([]), plt.yticks([])
#     plt.subplot(153), plt.imshow(image_h), plt.title("H = 100"), plt.xticks([]), plt.yticks([])
#     plt.subplot(154), plt.imshow(image_v), plt.title("V = +20"), plt.xticks([]), plt.yticks([])
#     plt.subplot(155), plt.imshow(image_s), plt.title("S = +150"), plt.xticks([]), plt.yticks([])

#     plt.show()

# method 2

for mask_name, image_name in zip(mask_names, image_names):
    
    mask = cv2.imread(os.path.join(mask_dir, mask_name))
    img = cv2.imread(os.path.join(image_dir, image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    intermediate = np.copy(img)
    intermediate[(mask==255).all(-1)] = [0,0,255]
    alpha = 0.8
    new_image = cv2.addWeighted(intermediate, 1-alpha, img, alpha,0)
    
    plt.figure()
    plt.subplot(131), plt.imshow(img), plt.title("Original image"), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(intermediate), plt.title("Intermediate image"), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(new_image), plt.title("Final result, alpha = 0.8"), plt.xticks([]), plt.yticks([])
    plt.show()
