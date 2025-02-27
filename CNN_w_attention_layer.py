import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
import logging
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from dataset_generation_from_img_and_labels import create_pokemon_card_dataset

tf.get_logger().setLevel(logging.ERROR)



# Test inference function
def test_model_inference(dataset, num_samples=5):
    print("\nTesting model on sample images:")
    for images, labels in dataset.take(1):
        for i in range(min(num_samples, len(images))):
            pred = model.predict(tf.expand_dims(images[i], 0))[0]
            top_indices = np.argsort(pred)[-2:][::-1]
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


# Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        scores = tf.matmul(x, self.W)
        attention_weights = tf.nn.softmax(scores, axis=1)
        attended_features = tf.multiply(x, attention_weights)
        return attended_features

    def compute_output_shape(self, input_shape):
        return input_shape


# Paths and constants
image_folder = "card_images_low_png_aug/card_images_BaseSet"
json_folder = "annotations_dir/labels_BaseSet"
IMG_SIZE = (224, 224)

# Create datasets
train_dataset, val_dataset, NUM_CLASSES, index_to_label = create_pokemon_card_dataset(
    image_folder=image_folder,
    json_folder=json_folder,
    batch_size=64,
    img_size=IMG_SIZE
)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])


# Define custom CNN model
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

    x = AttentionLayer()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


# Create and compile model
model = build_custom_cnn(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
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

# Train model
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Test some predictions
test_model_inference(val_dataset)

# Save model in TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('pokemon_card_classifier_custom.tflite', 'wb') as f:
    f.write(tflite_model)

print("Custom model saved as pokemon_card_classifier_custom.tflite")