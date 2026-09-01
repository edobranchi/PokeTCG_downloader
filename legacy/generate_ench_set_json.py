
# ⚠️ DEPRECATED — legacy per-set enrichment/merge step, superseded by
#    refresh_drifted_sets (invoked by `python pipeline.py metadata`), which
#    enriches rarity/variants/types directly from /cards/{id}. Kept for history;
#    do not add new callers. See ../README_PIPELINE.md.

#Aggiorna usando "image_labels" il json di ogni set in metadatadir, aggiungendo per ogni carta rarita
#e variazioni

import glob
import json
import os

# Define the paths to the folders
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "..", "assets")

first_folder = os.path.join(assets_dir, "metadata_dir")  # Folder containing the JSON files
second_folder = os.path.join(assets_dir, "image_labels")  # Folder containing the label files


# Function to load the card information from the second folder.
# Matches by card_id prefix so card name differences / special chars never cause misses.
def get_card_info_from_second_folder(card_id, card_name):
    pattern = os.path.join(second_folder, f"labels_{card_id}_*.json")
    matches = glob.glob(pattern)

    if not matches:
        return None, None, None

    label_filepath = matches[0]  # card_id is unique — only one match expected
    with open(label_filepath, 'r') as file:
        label_data = json.load(file)
    return label_data.get('variants', None), label_data.get('rarity', None), label_data.get('types', None)


if __name__ == "__main__":
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

                variants, rarity, types= get_card_info_from_second_folder(card_id, card_name)

                # Append variants and rarity to the card
                if variants is not None and rarity is not None:
                    card['variants'] = variants
                    card['rarity'] = rarity
                    card['types'] = types

            # Save the updated JSON data back to the file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)

            print(f"Updated {filename}")
