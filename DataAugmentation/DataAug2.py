import os
import cv2
import imgaug.augmenters as iaa

# Define the augmentation pipeline
aug_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip horizontally with 50% probability
    iaa.Affine(rotate=(-10, 10), scale=(0.8, 1.2)),  # Rotate and scale images
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),# Sharpen the image 
    iaa.LinearContrast((0.75, 1.5)),
    iaa.GaussianBlur(sigma=(0, 1.0)) # Apply Gaussian blur
    
])

# Define the root directory
root_dir = r"C:\Users\KUMGAUR\Downloads\DataAugmentation\Image"

# Loop through all the subdirectories and process each image
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Check if the file is an image file
        if file.endswith(('jpg', 'jpeg', 'png', 'bmp')):
            # Read the image
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path)
            # Apply the augmentation pipeline multiple times to create new images
            for i in range(16):  # Change the range to create more or fewer augmented images
                augmented_img = aug_pipeline.augment_image(img)
                # Create a new filename with a suffix indicating the augmentation technique
                new_file = os.path.splitext(file)[0] + f'_aug{i+1}' + os.path.splitext(file)[1]
                new_path = os.path.join(subdir, new_file)
                # Save the augmented image with the new filename
                cv2.imwrite(new_path, augmented_img)
