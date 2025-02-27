import tensorflow as tf
import os
import glob
import numpy as np
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import json


def create_pokemon_card_dataset(image_folder, json_folder, batch_size=64, img_size=(224, 224), train_split=0.8,
                                seed=42):
    """
    Creates training and validation datasets for Pokémon card classification.

    Args:
        image_folder (str): Path to folder containing PNG images
        json_folder (str): Path to folder containing JSON label files
        batch_size (int): Batch size for training
        img_size (tuple): Target image size (height, width)
        train_split (float): Proportion of data for training (0-1)
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_dataset, val_dataset, num_classes, index_to_label)
    """

    # Get all image paths
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))

    # Function to extract labels from JSON
    def get_label_from_json(image_path):
        image_name = os.path.basename(image_path).replace("_low_aug", "").rsplit("_", 1)[0]
        json_path = os.path.join(json_folder, "labels_" + image_name + ".json")
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                return data["name"]
        except FileNotFoundError:
            print(f"No labels found for {json_path}")
            return None

    # Generate and filter labels
    labels = [get_label_from_json(img_path) for img_path in image_paths]
    valid_pairs = [(img, lbl) for img, lbl in zip(image_paths, labels) if lbl is not None]

    if len(valid_pairs) == 0:
        raise ValueError("No valid labeled images found")

    image_paths, labels = zip(*valid_pairs)

    # Create label mappings
    unique_labels = sorted(set(labels))
    label_to_index = {name: i for i, name in enumerate(unique_labels)}
    index_to_label = {i: name for name, i in label_to_index.items()}
    numeric_labels = [label_to_index[label] for label in labels]
    one_hot_labels = tf.keras.utils.to_categorical(numeric_labels, num_classes=len(unique_labels))

    # Shuffle data
    indices = np.arange(len(image_paths))
    np.random.seed(seed)
    np.random.shuffle(indices)

    shuffled_paths = [image_paths[i] for i in indices]
    shuffled_labels = [one_hot_labels[i] for i in indices]

    # Split into train and validation
    train_size = int(train_split * len(shuffled_paths))
    train_paths = shuffled_paths[:train_size]
    train_labels = shuffled_labels[:train_size]
    val_paths = shuffled_paths[train_size:]
    val_labels = shuffled_labels[train_size:]

    # Image loading function
    def load_image(image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = preprocess_input(img)
        return img, label

    # Create datasets
    train_dataset = (tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
                     .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
                     .shuffle(buffer_size=1000)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = (tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
                   .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))

    # Print dataset info
    print(f"Total unique cards: {len(unique_labels)}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Class distribution (first 5): {list(label_to_index.items())[:5]}")

    return train_dataset, val_dataset, len(unique_labels), index_to_label


# Example usage:
if __name__ == "__main__":
    image_folder = "card_images_low_png_aug/card_images_BaseSet"
    json_folder = "annotations_dir/labels_BaseSet"

    train_ds, val_ds, num_classes, index_to_label = create_pokemon_card_dataset(
        image_folder=image_folder,
        json_folder=json_folder,
        batch_size=64,
        img_size=(224, 224),
        train_split=0.8,
        seed=42
    )