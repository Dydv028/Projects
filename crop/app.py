import os
import json
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('d.h5')

# Load class indices from JSON file
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the class_indices dictionary for easier lookup
class_names = {index: name for name, index in class_indices.items()}

# Define the function to prepare the image and make predictions
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match the input size of the model
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for MobileNetV2
    return img_array

# Print the class indices and class names for debugging
print(f"Class indices: {class_indices}")
print(f"Class names: {class_names}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the file is in the request
        if "file" not in request.files:
            return {"error": "No file part in the request"}, 400
        
        file = request.files["file"]
        
        # Check if a file was selected
        if file.filename == "":
            return {"error": "No file selected"}, 400
        
        # Save the uploaded image temporarily
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        img_path = os.path.join("uploads", file.filename)
        file.save(img_path)
        
        # Prepare the image and make predictions
        img_array = prepare_image(img_path)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names.get(predicted_class_index, "Unknown")
        
        # Return the prediction as a JSON response
        return {"prediction": predicted_class_name}
    
    # Render the HTML template for GET requests
    return render_template("index.html")
        
              

if __name__ == "__main__":
    app.run(debug=True)
