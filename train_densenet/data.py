
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils import data
from matplotlib import pyplot as plt
from data_augmentation import *

def generate_paths(base_path):
    paths_list = []

    names = sorted(os.listdir(base_path))
    
    for name in names:
        if name.split(".")[1] == 'txt':
            continue
        else:
            full_path = os.path.join(base_path + name)
            paths_list.append(full_path)

    return paths_list

def crop_image_and_mask(image, mask, margin):
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_box = None
    max_area = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        x -= margin
        y -= margin
        w += 2 * margin
        h += 2 * margin
        
        area = w * h
        
        if area > max_area:
            max_area = area
            largest_box = (x, y, w, h)

    if largest_box is not None:
        x, y, w, h = largest_box
        x = max(0, x)
        y = max(0, y)
        w = min(w, mask.shape[1] - x)
        h = min(h, mask.shape[0] - y)
        
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
    return cropped_image, cropped_mask




class HairSegmentationDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, resize_size, transforms = None):

        self.image_paths = generate_paths(image_dir)
        self.mask_paths = generate_paths(mask_dir)
        self.transforms = transforms
        self.resize_size = resize_size

    def __getitem__(self,index):

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        # _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        mask = mask[:,:,0]
        # image, mask = crop_image_and_mask(image, mask, margin=10)


        image = cv2.resize(image,(self.resize_size,self.resize_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask,(self.resize_size,self.resize_size), interpolation=cv2.INTER_AREA)

        if self.transforms is not None:
        # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)

        normalize = transforms.Normalize(mean=[0.4302, 0.4783, 0.5437],
                                  std=[0.2992, 0.2968, 0.2975])
        
        image = normalize(image)

        return (image, mask)


    def __len__(self):
        return len(self.mask_paths)


class ValidationDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, transforms = None):

        self.image_paths = generate_paths(image_dir)
        self.mask_paths = generate_paths(mask_dir)
        self.transforms = transforms

    def __getitem__(self,index):

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        mask = mask[:,:,0]
        # _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        # image, mask = crop_image_and_mask(image, mask, margin=10)

 
        if self.transforms is not None:
        # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)
  
        normalize = transforms.Normalize(mean=[0.4302, 0.4783, 0.5437],
                                  std=[0.2992, 0.2968, 0.2975])

        image = normalize(image)

        return (image, mask)


    def __len__(self):
        return len(self.mask_paths)

class AugumentationDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, resize_size, transforms = None):

        self.image_paths = generate_paths(image_dir)
        self.mask_paths = generate_paths(mask_dir)
        self.transforms = transforms
        self.resize_size = resize_size

    def __getitem__(self,index):

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        # _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        mask = mask[:,:,0]
        # image, mask = crop_image_and_mask(image, mask, margin=10)

        prob = random.uniform(0,1)
        image, mask = rotate(image, mask)
        

        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

        image = cv2.resize(image,(self.resize_size,self.resize_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask,(self.resize_size,self.resize_size), interpolation=cv2.INTER_AREA)

        if self.transforms is not None:
        # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)

        normalize = transforms.Normalize(mean=[0.4302, 0.4783, 0.5437],
                                  std=[0.2992, 0.2968, 0.2975])
        
        image = normalize(image)

        return (image, mask)


    def __len__(self):
        return len(self.mask_paths)