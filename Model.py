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

#TODO: Salvare le labels in un txt

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
NUM_CLASSES = len(unique_labels)


# Load MobileNetV3 as the base model
base_model = keras.applications.MobileNetV3Large(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

# Freeze base model initially, then unfreeze some layers for fine-tuning
base_model.trainable = False

# Build the model with functional API for more flexibility
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = base_model(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)  # Increased dropout to prevent overfitting
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)



optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.000001
)

# First round of training with frozen base model
print("Initial training with frozen base model...")

history_initial = model.fit(
    train_dataset,
    epochs=40,
    validation_data=val_dataset,
    callbacks=[early_stopping,reduce_lr]
)

test_model_inference(val_dataset,num_samples=10)

# # Fine-tuning: Unfreeze some layers of the base model
# print("Fine-tuning with partially unfrozen base model...")
# base_model.trainable = True
# fine_tune_at = 20  # Unfreeze the last 100 layers
# for layer in base_model.layers[:-fine_tune_at]:
#     layer.trainable = False
#
# # #Recompile with lower learning rate for fine-tuning
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# #Second round of training with fine-tuning
# history_fine_tune = model.fit(
#     train_dataset,
#     epochs=20,
#     validation_data=val_dataset,
#     callbacks=[early_stopping,reduce_lr]
# )

# Evaluate the model
print("Evaluating on validation set...")
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Save the model in TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save model
with open('pokemon_card_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved as pokemon_card_classifier.tflite")

# After saving the TFLite model, add this code:

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="pokemon_card_classifier.tflite")
interpreter.allocate_tensors()

# Import for image loading in TFLite function
# Example usage
image_path = 'test_images/hit.png'
image = useModel.preprocess_image_tflite(image_path)
# Example usage
useModel.predict_tflite(image,index_to_label)


