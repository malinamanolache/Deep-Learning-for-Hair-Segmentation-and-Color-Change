import os
from PIL import Image

# Path to the folder containing the original uncropped images
original_images_dir = "/shared_storage/manolac/dataset/yolo_unet_preds/bounding_boxes"

# Path to the folder containing the predicted cropped masks
cropped_masks_dir = "/shared_storage/manolac/dataset/yolo_unet_preds/unet_preds"

# Path to the folder where you want to save the resized masks
resized_masks_dir = "/shared_storage/manolac/dataset/yolo_unet_preds/final_masks"

# Iterate over the cropped masks
for mask_file in os.listdir(cropped_masks_dir):
    if mask_file.endswith(".png"):  # Adjust the file extension if needed
        # Load the cropped mask
        mask_path = os.path.join(cropped_masks_dir, mask_file)
        mask = Image.open(mask_path)

        # Load the corresponding original image
        image_name = os.path.splitext(mask_file)[0] + ".jpg"  # Adjust the file extension if needed
        image_path = os.path.join(original_images_dir, image_name)
        image = Image.open(image_path)

        # Get the dimensions of the original image
        original_width, original_height = image.size

        # Get the bounding box coordinates from the corresponding .txt file
        txt_file = os.path.splitext(mask_file)[0] + ".txt"
        txt_path = os.path.join(original_images_dir, txt_file)
        with open(txt_path, "r") as f:
            lines = f.readlines()

        # Sort the bounding boxes by their y-coordinate (ascending order)
        sorted_boxes = sorted(lines, key=lambda line: float(line.split()[2]))

        # Choose the bounding box closest to the top of the image
        top_box = sorted_boxes[0]

        # Extract the bounding box coordinates
        class_id, x, y, width, height = map(float, top_box.split())

        # Calculate the coordinates of the bounding box in the original image
        x_min = int((x - width / 2) * original_width)
        y_min = int((y - height / 2) * original_height)
        x_max = int((x + width / 2) * original_width)
        y_max = int((y + height / 2) * original_height)

        # Calculate the dimensions of the resized mask
        resized_width = x_max - x_min
        resized_height = y_max - y_min

        # Resize the mask to the size of the original image
        resized_mask = mask.resize((resized_width, resized_height), Image.NEAREST)

        # Create a new image with the size of the original image and fill it with black
        final_mask = Image.new("L", (original_width, original_height), 0)

        # Calculate the position to paste the resized mask in the final image
        paste_x = x_min
        paste_y = y_min

        # Paste the resized mask onto the final image
        final_mask.paste(resized_mask, (paste_x, paste_y))

        # Save the resized mask
        resized_mask_path = os.path.join(resized_masks_dir, mask_file)
        final_mask.save(resized_mask_path)
