
from networks.unet_aug import Unet_augmentation
from networks.unet import Unet
from networks.unet_dropout import Unet_dropout
from networks.unet_vgg import Unet_VGG, VGG
import torch
import os
from torchvision import transforms
import cv2
import numpy as np
from inference_yolo import infer_yolo


def predict(image, model_name):
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    height, width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    if model_name == "U-Net":
        model = Unet()
        model.to(device)
        model.load_state_dict(torch.load("D:\\MALY\\final_models_licenta\\unet_BCEloss_143.pth"))
    elif model_name == "U-Net dropout":
        model = Unet_dropout()
        model.to(device)
        model.load_state_dict(torch.load("D:\\MALY\\final_models_licenta\\UNET_dropout_191.pth"))
    elif model_name == "U-Net data augmentation":
        model = Unet_augmentation()
        model.to(device)
        model.load_state_dict(torch.load("D:\\MALY\\final_models_licenta\\UNET_data_augmentation_441.pth"))
    elif model_name == "U-Net pretrained VGG":
        vgg = VGG()
        model = Unet_VGG(vgg)
        model.to(device)
        model.load_state_dict(torch.load("D:\\MALY\\final_models_licenta\\U-Net_pretrained_177.pth"))
    elif model_name == "YOLOv4 + U-Net":
        # detect
        image, box_coordinates = infer_yolo(image)
        cropped_height, cropped_width,_ = image.shape
        model = Unet_augmentation()
        model.to(device)
        model.load_state_dict(torch.load("D:\\MALY\\final_models_licenta\\U-Net_cropped_209.pth"))

    model.eval()
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    normalize = transforms.Normalize(mean=[0.4302, 0.4783, 0.5437],
                                    std=[0.2992, 0.2968, 0.2975])
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    image = to_pil(image)
    image = to_tensor(image)
    image = normalize(image)
    image = image[None, :]
    image = image.to(device)
    prediction = model(image)
    prediction = prediction.cpu().detach().numpy()
    prediction = np.squeeze(prediction, axis = 0)

    prediction = (prediction > 0.5).astype(np.uint8)
    prediction = np.moveaxis(prediction, 0,-1)
    prediction = prediction*255
    if model_name == "YOLOv4 + U-Net":
        x, y, width_b, height_b = box_coordinates
        prediction = cv2.resize(prediction, (cropped_width, cropped_height), interpolation=cv2.INTER_AREA)       
        # Calculate the coordinates of the bounding box in the original image
        x_min = int((x - width_b / 2) * width)
        y_min = int((y - height_b / 2) * height)
        x_max = int((x + width_b / 2) * width)
        y_max = int((y + height_b / 2) * height)

        # Calculate the dimensions of the resized mask
        resized_width = x_max - x_min
        resized_height = y_max - y_min

        # Resize the mask to the size of the original image
        resized_mask = cv2.resize(prediction, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

        # Create a new image with the size of the original image and fill it with black
        final_mask = np.zeros((height, width), dtype=np.uint8)

        # Calculate the position to paste the resized mask in the final image
        paste_x = x_min
        paste_y = y_min

        # Paste the resized mask onto the final image
        final_mask[paste_y:paste_y+resized_height, paste_x:paste_x+resized_width] = resized_mask
        prediction = final_mask
    else:
        prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_AREA)

    print('Done')
    print(prediction.shape)

    return prediction

