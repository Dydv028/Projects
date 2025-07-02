from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "animal_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse class indices to map predictions
class_names = {v: k for k, v in class_indices.items()}

# Define image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size to match your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"})
    
    try:
        # Save and process image
        file_path = "temp.jpg"
        file.save(file_path)
        img_array = preprocess_image(file_path)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_label = class_names.get(predicted_class, "Unknown")

        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)