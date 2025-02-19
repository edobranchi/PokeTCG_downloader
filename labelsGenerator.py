import json
import os
import requests
from tqdm import tqdm

def generateLabels(setId, setName):
    setName=setName.replace(" ","")
    # Apre la cartella metadata e legge il file del set passato
    metadata_dir = "metadata_dir"
    file_path = os.path.join(metadata_dir, f"cards_metadata_{setName}.json")
    with open(file_path, "r") as file:
        cards = json.load(file)

    # Crea la cartella "annotations_dir"
    annotations_dir = "annotations_dir"
    os.makedirs(annotations_dir, exist_ok=True)

    # Crea la subfolder specifica del set
    annotations_dir_set = os.path.join(annotations_dir, f"labels_{setName}")
    annotations_dir_set = ''.join(annotations_dir_set.split())
    os.makedirs(annotations_dir_set, exist_ok=True)


    skipped_count = 0
    downloaded_count = 0

    # Genera le annotations per ogni carta
    for card in tqdm(cards['cards'], desc=f"Generating {setName} labels", unit="card"):
        #Estrae l'id della carta
        cardId = card["id"]

        # Generatione url e GET request per la singola carta
        API_URL_SINGLE_CARD = f"https://api.tcgdex.net/v2/en/cards/{cardId}"
        response = requests.get(API_URL_SINGLE_CARD)
        card_obtained = response.json()


        try:
            #estraggo id e nome della carta
            card_id = card_obtained["id"]
            card_name = card_obtained["name"].replace(" ","")

            #sostuisco il campo image della carta ottenuta dalla request, col percorso locale dell'immagine scaricata
            card_obtained["image"]=f"card_images_high_png/card_images_{setName}/{card_id}_{card_name}_high.png".replace(" ","")

            #rimuovo gli spazi dal campo image
            card_obtained["image"]=''.join(card_obtained["image"].split())

            #creo il file JSON della singola carta nella cartella del set a cui appartiene
            file_path = os.path.join(annotations_dir_set, f"labels_{card_id}_{card_name}.json")

        except Exception as e:
            print(f"Failed to create label ,id not present")

        # se il file esiste già skip
        if os.path.exists(file_path):
            skipped_count += 1
            continue

        try:
            # salva le annotazioni nel JSON appena creat0
            with open(file_path, "w") as json_file:
                json.dump(card_obtained, json_file, indent=4)
            downloaded_count += 1

        except Exception as e:
            print(f"Failed to create label for {card_id}: {e}")

    #Stampa recap
    print(f"\nLabel Generation Summary for {setName}:")
    print(f"Total cards in set: {len(cards['cards'])}")
    print(f"Successfully generated labels: {downloaded_count}")
    print(f"Skipped (already existing): {skipped_count}")