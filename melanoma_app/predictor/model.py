
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

classes = ['melanoma', 'non_melanoma']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "final_model")

print("Loading model...")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]
print("Model loaded successfully")


def predict_image(img_path):
    print("Processing:", img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # 🔹 Convert to tensor
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # 🔹 Run inference
    output = infer(input_tensor)

    # 🔹 Get prediction tensor
    prediction = list(output.values())[0].numpy()

    print("Raw prediction:", prediction)

    idx = int(np.argmax(prediction))
    label = classes[idx]
    confidence = float(prediction[0][idx])

    return label, confidence