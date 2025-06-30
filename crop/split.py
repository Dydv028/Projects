import os
import shutil
import random

# Set paths
data_dir = "data"  # Change if needed
output_dir = "split_data"
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Define output directories
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Create directories
for dir_path in [train_dir, val_dir, test_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Iterate through categories
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    
    if os.path.isdir(category_path):  # Ensure it's a folder
        for subcategory in os.listdir(category_path):
            subcategory_path = os.path.join(category_path, subcategory)
            
            if os.path.isdir(subcategory_path):  # Ensure it's a folder
                images = [img for img in os.listdir(subcategory_path) if img.endswith(('.png', '.jpg', '.jpeg','.JPG'))]
                random.shuffle(images)

                # Calculate split indices
                total_images = len(images)
                train_idx = int(total_images * train_ratio)
                val_idx = train_idx + int(total_images * val_ratio)

                # Split the images
                train_images = images[:train_idx]
                val_images = images[train_idx:val_idx]
                test_images = images[val_idx:]

                # Function to copy images
                def copy_images(image_list, dest_folder):
                    dest_path = os.path.join(dest_folder, category, subcategory)
                    os.makedirs(dest_path, exist_ok=True)
                    for img in image_list:
                        shutil.copy(os.path.join(subcategory_path, img), os.path.join(dest_path, img))

                copy_images(train_images, train_dir)
                copy_images(val_images, val_dir)
                copy_images(test_images, test_dir)

print("Dataset split complete!")
