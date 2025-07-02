import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset directory
data_dir = "data"

# Load training data
datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Print class indices assigned during training
print("✅ Class indices assigned during training:", train_generator.class_indices)


# Define paths
base_dir = "split_data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load images from folders
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Get number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes detected: {num_classes}")

# Load MobileNetV2 as a pretrained model
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model layers

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Pooling Layer
x = Dense(512, activation="relu")(x)  # Fully Connected Layer
x = Dropout(0.4)(x)  # Dropout
x = Dense(256, activation="relu")(x)  # Fully Connected Layer
x = Dropout(0.3)(x)  # Dropout Layer
output_layer = Dense(num_classes, activation="softmax")(x)  # Output Layer

# Define the model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Save trained model
model.save("animal_classifier.h5")
print("✅ Model training complete. Model saved as animal_classifier.h5")

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"📊 Test Accuracy: {test_acc * 100:.2f}%")
