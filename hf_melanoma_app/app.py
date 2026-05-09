import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.saved_model.load("final_model")

infer = model.signatures["serving_default"]

classes = ['melanoma', 'non_melanoma']


def predict(img):

    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    input_tensor = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32
    )

    output = infer(input_tensor)

    prediction = list(output.values())[0].numpy()

    idx = int(np.argmax(prediction))

    label = classes[idx]

    confidence = float(prediction[0][idx])

    return f"{label} ({confidence:.4f})"


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Melanoma Detection AI",
    description="Upload a skin lesion image"
)

interface.launch()