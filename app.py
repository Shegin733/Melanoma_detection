from flask import Flask, request, render_template, send_from_directory, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os
import uuid

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels (must match training order)
classes = ['melanoma', 'non_melanoma']


# 🔹 Rebuild model architecture
def build_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=x)
    return model


# 🔹 Load weights
model = build_model()
model.load_weights("model.weights.h5")


# 🔹 Prediction function
def predict_image(img_path):
    print("Processing:", img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    print("Raw prediction:", prediction)

    idx = int(np.argmax(prediction))
    label = classes[idx]
    confidence = float(prediction[0][idx])

    return label, confidence


# 🔹 Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# 🔹 Main route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            return "No file uploaded"

        # Unique filename (avoids overwrite/cache issues)
        filename = str(uuid.uuid4()) + ".jpg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        print("Saved:", file_path)

        label, confidence = predict_image(file_path)

        return render_template(
            "index.html",
            prediction=label,
            confidence=round(confidence, 4),
            image_url=url_for('uploaded_file', filename=filename)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)