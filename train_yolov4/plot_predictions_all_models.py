import os
import matplotlib.pyplot as plt

# Directory paths
directories = ['D:\\MALY\\hair_segmentation_dataset\\test\\image', 
               'D:\\MALY\\hair_segmentation_dataset\\test\\GT_png', 
               'D:\\MALY\\final_preds\\unet', 
               'D:\\MALY\\final_preds\\unet_dropout', 
               'D:\\MALY\\final_preds\\unet_data_augm', 
               'D:\\MALY\\final_preds\\unet_pretrained', 
               'D:\\MALY\\final_preds\\densenet_preds_corect', 
               'D:\\MALY\\final_preds\\yolo']
output_directory = 'plots'  # Directory to save the plots

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get the list of images in the original directory (assuming all directories have the same number of images)
original_dir = directories[0]
image_files = sorted(os.listdir(original_dir))

# Iterate over the images
for image_file in image_files:
    # Create a figure with subplots
    fig, axs = plt.subplots(1, len(directories), figsize=(15, 5))  # Adjust the figsize as per your preference

    # Plot images from each directory
    for i, directory in enumerate(directories):
        # Determine the corresponding image file
        dir_name = directory.split('\\')[-1]
        
        if dir_name == 'image':
            # For the original directory, use .jpg extension and remove any suffixes
            image_path = os.path.join(directory, image_file)
        elif dir_name == 'GT_png':
            # For the gt directory, replace the suffix with .png
            image_path = os.path.join(directory, image_file.replace('-org.jpg', '-gt.png'))
        else:
            # For the other directories, replace the suffix with .png and change the directory name
            image_path = os.path.join(directory, image_file.replace('-org.jpg', '-org.png'))

        # Read the image
        image = plt.imread(image_path)

        # Plot the image
        if dir_name == "image":
            axs[i].imshow(image)
        else:
            axs[i].imshow(image, cmap = "gray")
        axs[i].axis('off')

        # Set the title for each subplot as the directory name
        

    # Save the plot as PNG file
    output_file = os.path.join(output_directory, image_file.replace('-org.jpg', ''))
    plt.savefig(output_file, dpi=400, bbox_inches='tight')

    # Close the plot
    plt.close()

    print(f"Plot saved: {output_file}")