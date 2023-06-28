import cv2
import os
from matplotlib import pyplot as plt
import tqdm

blended_dir = "./blended/"
not_blended_dir = "./not_blended/"
median_blur_dir = "./median_blur_3x3/"
comparison_dir = "./edge_fading_comparison/"

image_names = sorted(os.listdir(blended_dir))

for name in tqdm.tqdm(image_names):
    blended = cv2.imread(blended_dir+name)
    not_blended = cv2.imread(not_blended_dir+name)
    # median_blur_mask = cv2.imread(median_blur_dir+name)

    plt.figure()
    plt.subplot(121), plt.imshow(not_blended[:,:,::-1]), plt.title("Without edge fading", fontsize = 7), plt.xlim(0, 150), plt.ylim(400, 500), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blended[:,:,::-1]), plt.title("With edge fading", fontsize = 7), plt.xlim(0, 150), plt.ylim(400, 500), plt.xticks([]), plt.yticks([])
    # plt.subplot(133), plt.imshow(median_blur_mask, cmap = "gray"), plt.title("Median Blur", fontsize = 7), plt.xlim(0, 150), plt.ylim(400, 500), plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(comparison_dir+name, dpi=400, bbox_inches='tight')