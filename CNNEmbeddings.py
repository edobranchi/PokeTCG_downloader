import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Lambda
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

#TODO: dhn overfitta con 100 immagini per classe, provare meno


# Paths
image_folder = "card_images_low_png_aug/card_images_BaseSet"
json_folder = "annotations_dir/labels_BaseSet"

# Load image paths
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))

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

# Filter out invalid labels
valid_pairs = [(img, lbl) for img, lbl in zip(image_paths, labels) if lbl is not None]
if len(valid_pairs) == 0:
    raise ValueError("No valid labeled images found.")

image_paths, labels = zip(*valid_pairs)

# Convert labels to numeric
unique_labels = sorted(set(labels))
label_to_index = {name: i for i, name in enumerate(unique_labels)}
index_to_label = {i: name for name, i in label_to_index.items()}
numeric_labels = [label_to_index[label] for label in labels]
one_hot_labels = tf.keras.utils.to_categorical(numeric_labels, num_classes=len(unique_labels))

# Preprocessing function
def load_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    return img, label

# Shuffle dataset
indices = np.arange(len(image_paths))
np.random.seed(42)
np.random.shuffle(indices)

shuffled_paths = [image_paths[i] for i in indices]
shuffled_labels = [one_hot_labels[i] for i in indices]

# Split into training and validation
train_size = int(0.8 * len(shuffled_paths))
train_paths = shuffled_paths[:train_size]
train_labels = shuffled_labels[:train_size]
val_paths = shuffled_paths[train_size:]
val_labels = shuffled_labels[train_size:]

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_image).batch(64).shuffle(buffer_size=1000)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(load_image).batch(64)

# Embedding Model
def create_embedding_model(input_shape=(224, 224, 3), embedding_dim=256):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Dropout(0.3)(x)  # Try dropout rates like 0.3, 0.5, or 0.7
    outputs = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return models.Model(inputs, outputs, name="PokemonCardEmbeddingModel")

model = create_embedding_model()

# Triplet Loss Function
def triplet_loss(margin=0.2):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

# Compile the model
model.compile(optimizer='adam', loss=triplet_loss(),metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Save embeddings for later similarity search
def generate_embeddings(model, dataset):
    embeddings = []
    labels = []
    for images, lbls in dataset:
        emb = model.predict(images)
        embeddings.append(emb)
        labels.append(lbls)
    return np.vstack(embeddings), np.vstack(labels)

embeddings, embedding_labels = generate_embeddings(model, val_dataset)
np.save("embeddings.npy", embeddings)
np.save("embedding_labels.npy", embedding_labels)








######TEST#####


# Load saved embeddings and labels
embeddings = np.load("embeddings.npy")
embedding_labels = np.load("embedding_labels.npy")

def get_image_embedding(image_path, model):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    embedding = model.predict(img)
    return embedding[0]  # Return embedding vector




def find_similar_card(embedding, embeddings, embedding_labels, top_k=3):
    similarities = cosine_similarity([embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # Top-k most similar

    results = []
    for idx in top_indices:
        label_idx = np.argmax(embedding_labels[idx])  # Convert one-hot back to label index
        card_name = index_to_label[label_idx]
        similarity_score = similarities[idx]
        results.append((card_name, similarity_score))

    return results


# Path to a new card image for testing
test_image_path = "test_images/hit.png"

# Generate embedding
test_embedding = get_image_embedding(test_image_path, model)

# Find similar cards
similar_cards = find_similar_card(test_embedding, embeddings, embedding_labels)

# Display results
print("Top similar cards:")
for card_name, score in similar_cards:
    print(f"{card_name} (Similarity: {score:.4f})")




def show_similar_cards(query_image_path, similar_cards):
    plt.figure(figsize=(10, 5))

    # Display query image
    query_img = Image.open(query_image_path)
    plt.subplot(1, len(similar_cards) + 1, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis("off")

    # Display top similar cards
    for i, (card_name, _) in enumerate(similar_cards, 2):
        card_image_path = image_paths[labels.index(card_name)]
        img = Image.open(card_image_path)
        plt.subplot(1, len(similar_cards) + 1, i)
        plt.imshow(img)
        plt.title(card_name)
        plt.axis("off")

    plt.show()


# Visualize results
show_similar_cards(test_image_path, similar_cards)


