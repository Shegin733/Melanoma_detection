

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "final_model")

classes = ['melanoma', 'non_melanoma']

model = None
infer = None


def load_model():
    global model, infer

    if model is None:
        print("Loading model...")
        model = tf.saved_model.load(MODEL_PATH)
        infer = model.signatures["serving_default"]
        print("Model loaded")


def predict_image(img_path):
    load_model()

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    output = infer(input_tensor)

    prediction = list(output.values())[0].numpy()

    idx = int(np.argmax(prediction))

    label = classes[idx]
    confidence = float(prediction[0][idx])

    return label, confidence