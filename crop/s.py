import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the dataset directory and split folders
base_dir = "split_data"  # Update with your correct base directory
train_dir = os.path.join(base_dir, "train")

# Initialize the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255)

# Create a generator to access the class indices (train_generator)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Ensure this matches the input size used during training
    batch_size=32,
    class_mode='categorical'
)

# Retrieve and print the class indices
class_labels = train_generator.class_indices
print("Class Labels used in training:")
for class_name, class_index in class_labels.items():
    print(f"Class name: {class_name}, Class index: {class_index}")
