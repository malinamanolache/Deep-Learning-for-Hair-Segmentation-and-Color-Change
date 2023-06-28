from PIL import Image
import os
import tqdm

in_dir = "D:\MALY\hair_segmentation_dataset\\train\GT\\"
out_dir = "D:\MALY\hair_segmentation_dataset\\train\GT_png\\"

image_names = sorted(os.listdir(in_dir))

for image_name in tqdm.tqdm(image_names):
    image = Image.open(in_dir + image_name)
    name = image_name.split(".")[0]
    image.save(out_dir + name + ".png")