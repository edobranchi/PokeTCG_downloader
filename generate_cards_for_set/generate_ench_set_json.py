

#Aggiorna usando "image_labels" il json di ogni set in metadatadir, aggiungendo per ogni carta rarita
#e variazioni

import json
import os

# Define the paths to the folders
first_folder = "../assets/metadata_dir"  # Folder containing the JSON files
second_folder = "../assets/image_labels"  # Folder containing the label files


# Function to load the card information from the second folder
def get_card_info_from_second_folder(card_id, card_name):
    # Construct the filename for the corresponding label file
    card_name= card_name.replace(" ", "");
    label_filename = f"labels_{card_id}_{card_name}.json"
    label_filepath = os.path.join(second_folder, label_filename)

    if os.path.exists(label_filepath):
        with open(label_filepath, 'r') as file:
            label_data = json.load(file)
            return label_data.get('variants', None), label_data.get('rarity', None)
    return None, None


# Iterate through each file in the first folder
for filename in os.listdir(first_folder):
    file_path = os.path.join(first_folder, filename)

    if os.path.isfile(file_path) and filename.endswith('.json'):
        # Open and load the content of the first folder JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # For each card in the 'cards' list, find the corresponding label file in the second folder
        for card in data['cards']:
            card_id = card['id']
            card_name = card['name']

            # Get the variants and rarity from the second folder
            variants, rarity = get_card_info_from_second_folder(card_id, card_name)

            # Append variants and rarity to the card
            if variants is not None and rarity is not None:
                card['variants'] = variants
                card['rarity'] = rarity

        # Save the updated JSON data back to the file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Updated {filename}")
