import os
import json

train_dir = "split_data/train"  # change if needed

# Get all subfolders (class labels) in current order (as seen by the OS, not sorted)
class_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]

# Build class_indices based on their current order
class_indices = {label: idx for idx, label in enumerate(class_folders)}

# Save to JSON
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=2)

print("✅ class_indices.json created successfully.")
