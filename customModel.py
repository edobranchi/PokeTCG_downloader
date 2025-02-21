import tensorflow as tf
import json
import os
import glob
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
import logging
from tensorflow.keras.preprocessing import image as tf_image
import useModel
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input



tf.get_logger().setLevel(logging.ERROR)
# Paths to images and JSON files
image_folder = "card_images_low_png_aug/card_images_BaseSet"
json_folder = "annotations_dir/labels_BaseSet"

# Get all image paths
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
# Optional: Test inference on a few validation examples
def test_model_inference(dataset, num_samples=5):
    print("\nTesting model on sample images:")
    for images, labels in dataset.take(1):
        for i in range(min(num_samples, len(images))):
            pred = model.predict(tf.expand_dims(images[i], 0))[0]

            # Get top 2 predictions
            top_indices = np.argsort(pred)[-2:][::-1]  # Sort and get top 2 in descending order
            pred_idx = top_indices[0]
            second_pred_idx = top_indices[1]

            true_idx = np.argmax(labels[i])

            pred_name = index_to_label[pred_idx]
            second_pred_name = index_to_label[second_pred_idx]
            true_name = index_to_label[true_idx]

            confidence = pred[pred_idx] * 100
            second_confidence = pred[second_pred_idx] * 100

            print(f"Sample {i + 1}:")
            print(f"  Predicted: {pred_name} (Confidence: {confidence:.2f}%)")
            print(f"  Second choice: {second_pred_name} (Confidence: {second_confidence:.2f}%)")
            print(f"  Actual: {true_name}")
            print(f"  {'✓ Correct' if pred_idx == true_idx else '✗ Incorrect'}")
            print()

# Function to extract labels from JSON
def get_label_from_json(image_path):
    image_name = os.path.basename(image_path).replace("_low_aug", "").rsplit("_", 1)[0]
    json_path = os.path.join(json_folder, "labels_" + image_name + ".json")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            return data["name"]  # Extract card name
    except FileNotFoundError:
        print(f"No labels found for {json_path}")
        return None


# Generate labels for images
labels = [get_label_from_json(img_path) for img_path in image_paths]

# Remove None values (cases where labels were missing)
valid_pairs = [(img, lbl) for img, lbl in zip(image_paths, labels) if lbl is not None]
if len(valid_pairs) == 0:
    raise ValueError("No valid labeled images found")

image_paths, labels = zip(*valid_pairs)

# Convert labels to one-hot encoding
unique_labels = sorted(set(labels))  # Get unique Pokémon card names
label_to_index = {name: i for i, name in enumerate(unique_labels)}
index_to_label = {i: name for name, i in label_to_index.items()}  # Reverse mapping for later use
numeric_labels = [label_to_index[label] for label in labels]

# Convert to one-hot encoding
one_hot_labels = tf.keras.utils.to_categorical(numeric_labels, num_classes=len(unique_labels))


# Function to load and preprocess images
def load_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)  # Apply MobileNetV3 normalization
    return img, label

# Create datasets with proper shuffling
# First, shuffle with a fixed seed for reproducibility
indices = np.arange(len(image_paths))
np.random.seed(42)
np.random.shuffle(indices)

shuffled_paths = [image_paths[i] for i in indices]
shuffled_labels = [one_hot_labels[i] for i in indices]

# Create train/validation split
train_size = int(0.8 * len(shuffled_paths))
train_paths = shuffled_paths[:train_size]
train_labels = shuffled_labels[:train_size]
val_paths = shuffled_paths[train_size:]
val_labels = shuffled_labels[train_size:]

# Create separate datasets for train and validation
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_image).batch(64).shuffle(buffer_size=1000)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(load_image).batch(64)

# Print dataset info
print(f"Total unique cards: {len(unique_labels)}")
print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print(f"Class distribution (first 5): {list(label_to_index.items())[:5]}")

# Define data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])


# Image dimensions and number of classes
IMG_SIZE = (224, 224)
NUM_CLASSES = 102  # Matches your dataset

# Define a custom CNN model
def build_custom_cnn(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)  # Reduces dimensions before fully connected layers

    x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

# Create and compile the model
model = build_custom_cnn(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.000001
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_dataset,
    epochs=40,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Save the model in TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('pokemon_card_classifier_custom.tflite', 'wb') as f:
    f.write(tflite_model)

print("Custom model saved as pokemon_card_classifier_custom.tflite")
