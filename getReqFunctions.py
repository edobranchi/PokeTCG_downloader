import json
import os
import requests
import settings
from tqdm import tqdm


def singleSetRequest(setId, setName):
    #Crea la cartella "metadata_dir"
    metadata_dir = "metadata_dir"
    os.makedirs(metadata_dir, exist_ok=True)

    #GET request per ottenere il set passato
    API_URL_SINGLESET = f"https://api.tcgdex.net/v2/en/sets/{setId}"
    response = requests.get(API_URL_SINGLESET)

    if response.status_code == 200:
        cards = response.json()

        #Crea file json "cards_metadata_{Nome set}" con i metadati del set
        file_path = os.path.join(metadata_dir, f"cards_metadata_{setName}.json")
        with open(file_path, "w") as file:
            json.dump(cards, file, indent=4)

        #conta il numero di carte nel set e stampa
        number_of_card_in_set = len(cards["cards"])
        print(f"Downloaded metadata for {number_of_card_in_set} cards in the {setName} Set.")
    else:
        print(f"Failed to fetch data: {response.status_code} {response.json()}")

    #chiama la funzione di download sul set
    ImgDownload(setId, setName, metadata_dir)


def ImgDownload(setId, setName, metadata_dir):
    #apre il file di metadati del set generato da singleSetRequest()
    file_path = os.path.join(metadata_dir, f"cards_metadata_{setName}.json")
    with open(file_path, "r") as file:
        cards = json.load(file)

    #Crea la cartella "card_images_{qualità immagine}_{estensione immagine}"
    image_dir = f"./card_images_{settings.IMAGE_QUALITY}_{settings.IMG_EXTENSION}"
    os.makedirs(image_dir, exist_ok=True)

    # Crea la subfolder per il singolo set "card_images_{nome set}"
    image_dir_set=f"card_images_{setName}"
    image_dir_set = image_dir_set.replace(" ","")
    image_dir_path = os.path.join(image_dir, image_dir_set)
    os.makedirs(image_dir_path, exist_ok=True)


    skipped_count = 0
    downloaded_count = 0

    #ciclo sulle singole carte del set
    for card in tqdm(cards['cards'], desc=f"Downloading {setName} images", unit="card"):
        try:
            #Nei metadati della carta c'è il link all'immagine, ci appendo, qualità e estensione desiderata
            image_url = card["image"] + "/" + f"{settings.IMAGE_QUALITY}.{settings.IMG_EXTENSION}"

            if image_url:
                #annoto Id e nome della carta
                card_id = card.get("id")
                card_name = card.get("name")

                #Appende il percorso della cartella del set con ID, name, quality, and extension
                file_path = os.path.join(image_dir_path,
                                         f"{card_id}_{card_name}_{settings.IMAGE_QUALITY}.{settings.IMG_EXTENSION}")

                #Rimuove gli spazi nel nome del file
                file_path = ''.join(file_path.split())

                # Controlla se la carta esiste già nella cartella immagini e la salta in caso di esito positivo
                if os.path.exists(file_path):
                    skipped_count += 1
                    continue

                try:

                    #GET request per l'immagine della singola carta
                    img_data = requests.get(image_url).content

                    #Salva la carta
                    with open(file_path, "wb") as img_file:
                        img_file.write(img_data)
                    downloaded_count += 1

                except Exception as e:
                    print(f"Failed to download {card_id}: {e}")
        except Exception as e:
            print("no images skipping")

    # Stampa recap del download del singolo set
    print(f"\nDownload Summary for {setName}:")
    print(f"Total cards in set: {len(cards['cards'])}")
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"Skipped (already existing): {skipped_count}")