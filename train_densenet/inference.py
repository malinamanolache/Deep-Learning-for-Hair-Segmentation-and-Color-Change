from config import *
from densenet import Densenet_Unet

import torch
import os
import tqdm
from torchvision import transforms
from evaluate import *
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = Unet_VGG()
# model.to(device)

# Initialize model
# vgg = VGG()
# # temp = torchvision.models.vgg16(pretrained=True)
# vgg.load_state_dict(torch.load("/shared_storage/manolac/network_vgg/models/vgg16_pretrained.pth"), strict = False)
# # # freeze layers
# for param in vgg.parameters():
#     param.requires_grad = False
model = Densenet_Unet()
# model = Unet()
model.to(device)

model.load_state_dict(torch.load(MODEL_PATH))

model.eval()

normalize = transforms.Normalize(mean=[0.4302, 0.4783, 0.5437],
                                  std=[0.2992, 0.2968, 0.2975])
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

inference_samples = sorted(os.listdir(INPUT_DIR))

for sample in inference_samples:

    extension = sample.split(".")[-1]
    if extension == "txt":
        inference_samples.remove(sample)

print("save dir:", OUTPUT_DIR)
for sample in tqdm.tqdm(inference_samples):
    
    image = cv2.imread(os.path.join(INPUT_DIR, sample))
    h, w, c = image.shape
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    image = to_pil(image)
    image = to_tensor(image)
    image = normalize(image)
    image = image[None, :]
    image = image.to(device)
    prediction = model(image)
    # print(prediction.max())
    prediction, x = prepare_for_evaluation(prediction, prediction)
    prediction = np.moveaxis(prediction, 0,-1)
    prediction = prediction*255
    prediction = cv2.resize(prediction, (w,h), interpolation=cv2.INTER_AREA)
    name = sample.split(".")[0]

    cv2.imwrite(os.path.join(OUTPUT_DIR, name + ".png"), prediction)

