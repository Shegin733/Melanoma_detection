import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Load dataset (ensure 'dataset' folder is in the same directory)
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# Print class names
print("Class names:", dataset.class_names)

# Normalize pixel values
dataset = dataset.map(lambda x, y: (x / 255.0, y))

# Limit dataset size for faster local training
dataset = dataset.take(20)

# Load pretrained model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# Add classification layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(2, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(dataset, epochs=4)

# Save model
model.save("model.h5")

print("Model trained and saved")