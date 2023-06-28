
import os
import tqdm

orig_dir = "D:\MALY\hair_segmentation_dataset\\train\GT\\"
mask_dir = "D:\MALY\hair_segmentation_dataset\\train\GT\\"

original = sorted(os.listdir(orig_dir))
masks = sorted(os.listdir(mask_dir))

original_names = []

for orig in original:
    name = orig.split("-")[0]
    original_names.append(name)

for mask in tqdm.tqdm(masks):
    name = mask.split("-")[0]
    if name not in original_names:
        print(name)


