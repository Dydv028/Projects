import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import os
import json

# Load your class_indices.json
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Create inverse mapping for class_mode consistency
class_mode_mapping = {v: k for k, v in class_indices.items()}

# Set dataset directories
base_dir = "split_data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Image properties
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20

# Data augmentation for training
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

# Rescale only for val/test
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Custom class sorting function (match your class_indices.json order)
def sorted_classes(path):
    return sorted(
        [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))],
        key=lambda x: class_indices[x]
    )

# Create train, val, test generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=sorted_classes(train_dir),  # 👈 Enforces your order
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=sorted_classes(val_dir),
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=sorted_classes(test_dir),
    class_mode='categorical'
)

# Confirm the classes match your file
print("✅ Class Indices Used:")
print(train_generator.class_indices)

# Load pre-trained MobileNetV2
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet")
base_model.trainable = False

# Build custom model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(len(class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# Compile model
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

# Save the model
model.save("d.h5")
print("✅ Model saved as crop.h5")

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"📊 Test Accuracy: {test_acc * 100:.2f}%")
