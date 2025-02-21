import glob
import json
import os

import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


# Paths to images and JSON files
image_folder = "card_images_low_png_aug/card_images_BaseSet"
json_folder = "annotations_dir/labels_BaseSet"

# Get all image paths
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
# Optional: Test inference on a few validation examples


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
index_to_label = {i: name for name, i in label_to_index.items()}



# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='pokemon_card_classifier_custom.tflite')
interpreter.allocate_tensors()



def preprocess_image_tflite(image_path):
    img = tf_image.load_img(image_path, target_size=(224, 224))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Apply MobileNetV3 preprocessing
    return img_array.astype(np.float32)  # Ensure correct dtype




def predict_tflite(image,index_to_label):
    # Get tensor details (input and output tensors)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor data (input image)
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    print(f"Predicted class: {predicted_class}")
    # Map back to the class name
    predicted_label = index_to_label[predicted_class[0]]
    print(f"Predicted class: {predicted_label}")


if __name__ == "__main__":
    # Example image file path
    image_path = "./card_images_low_png/card_images_BaseSet/base1-3_Chansey_low.png"



    # Preprocess the image
    preprocessed_image = preprocess_image_tflite(image_path)

    # Perform prediction
    predict_tflite(preprocessed_image, index_to_label)
