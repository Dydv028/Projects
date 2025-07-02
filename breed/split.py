import os
import shutil
import random

# Paths
data_dir = "data"  
output_dir = "split_data"  

# Split ratios
split_ratio = {"train": 0.7, "val": 0.2, "test": 0.1}  

# Create train, val, test directories
for split in split_ratio.keys():
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# Function to split and copy images
def split_images(category, img_paths):
    random.shuffle(img_paths)
    total_images = len(img_paths)
    
    train_end = int(split_ratio["train"] * total_images)
    val_end = train_end + int(split_ratio["val"] * total_images)

    splits = {
        "train": img_paths[:train_end],
        "val": img_paths[train_end:val_end],
        "test": img_paths[val_end:]
    }

    for split, images in splits.items():
        split_cat_dir = os.path.join(output_dir, split, category)
        os.makedirs(split_cat_dir, exist_ok=True)
        for img in images:
            shutil.copy(img, os.path.join(split_cat_dir, os.path.basename(img)))

# Process each category
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if not os.path.isdir(category_path):
        continue  # Skip non-directory files

    image_paths = []

    # Check if category contains subfolders (Birds, Snake)
    if any(os.path.isdir(os.path.join(category_path, sub)) for sub in os.listdir(category_path)):
        for subfolder in os.listdir(category_path):
            sub_path = os.path.join(category_path, subfolder)
            if os.path.isdir(sub_path):
                image_paths.extend([os.path.join(sub_path, img) for img in os.listdir(sub_path) 
                                    if img.lower().endswith(('.jpg', '.jpeg', '.png','.JPG'))])
    else:
        image_paths = [os.path.join(category_path, img) for img in os.listdir(category_path) 
                       if img.lower().endswith(('.jpg', '.jpeg', '.png','.JPG'))]

    # Debugging prints
    print(f"Category: {category}, Images Found: {len(image_paths)}")

    if image_paths:
        split_images(category, image_paths)

print("✅ Dataset successfully split into train, val, and test sets!")
