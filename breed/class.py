import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
data_dir = "data"  # Change this if your dataset is in a different location
output_file = "class_indices.json"  # File to save class indices

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(rescale=1.0/255)

# Load dataset (only training set is needed to get class indices)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Change based on your model input size
    batch_size=32,
    class_mode='categorical'
)

# Get class indices
class_indices = train_generator.class_indices

# Save to a JSON file
with open(output_file, "w") as f:
    json.dump(class_indices, f, indent=4)

print(f"Class indices saved to {output_file}")
