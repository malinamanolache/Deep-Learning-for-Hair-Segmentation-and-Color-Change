import os
from matplotlib import pyplot as plt
import cv2

original_images_dir = 'D:\\MALY\\hair_segmentation_dataset\\test\\image\\' 
detections_dir = 'D:\\MALY\\predictions\\yolo_pretrained_vgg\\detections'
output_dir = 'D:\\MALY\\predictions\\yolo_pretrained_vgg\\original_and_detected'

original_names = sorted(os.listdir(original_images_dir))
detection_names = sorted(os.listdir(detections_dir))

for orig_name, detection_name in zip(original_names, detection_names):

    orig_img = cv2.imread(os.path.join(original_images_dir, orig_name))
    detection = cv2.imread(os.path.join(detections_dir, detection_name))
    plt.figure()
    plt.subplot(121)
    plt.imshow(orig_img[:,:,::-1]), plt.title("Original image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(detection[:,:,::-1]), plt.title("Detection"), plt.xticks([]), plt.yticks([])
    plt.savefig(os.path.join(output_dir, detection_name), dpi=400, bbox_inches='tight')
    plt.close()