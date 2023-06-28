import os
from PIL import Image

# Path to the directory containing the images and bounding box files
data_dir = "D:\\MALY\\predictions\\yolo\\detections\\"
out_dir = "D:\\MALY\\predictions\\yolo\\cropped_images\\" 
samples_dir = "D:\\MALY\\hair_segmentation_dataset\\test\\image\\"

# Iterate over the files in the directory
for file_name in os.listdir(data_dir):
    if file_name.endswith(".txt"):
        # Read the bounding box coordinates from the .txt file
        with open(os.path.join(data_dir, file_name), "r") as f:
            lines = f.readlines()
            sorted_boxes = sorted(lines, key=lambda line: float(line.split()[2]))
            
            # Choose the bounding box closest to the top of the image
            top_box = sorted_boxes[0]
            
            # Parse the coordinates
            class_id, x, y, width, height = map(float, top_box.split())

        # Load the corresponding image
        image_name = os.path.splitext(file_name)[0] + ".jpg"
        image_path = os.path.join(samples_dir, image_name)
        image = Image.open(image_path)

        # Calculate the coordinates for cropping
        image_width, image_height = image.size
        left = int((x - width / 2) * image_width)
        top = int((y - height / 2) * image_height)
        right = int((x + width / 2) * image_width)
        bottom = int((y + height / 2) * image_height)

        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))

        # Save the cropped image
        cropped_image.save(os.path.join(out_dir, image_name))
