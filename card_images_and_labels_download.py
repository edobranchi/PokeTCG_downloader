#Recupera tutti i set disponibili
import json
import os

import requests
from tqdm import tqdm

import settings

API_URL_ALLSETS= f"https://api.tcgdex.net/v2/en/sets"
response = requests.get(API_URL_ALLSETS)
sets=response.json()


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



if __name__ == "__main__":

    #Vuoto per scaricarli tutti altrimenti prendere dalla lista in basso
    setId_to_download = ""
    setName_to_download = ""

    if setId_to_download == "" and setName_to_download == "":
        #Cicla su ogni set disponibile
        for set in sets:
            setId=set["id"]
            setName= set["name"]
            print(setId, setName)
            #
            # #singleSetRequest(...) scarica e organizza in cartelle il singolo set
            # singleSetRequest(setId,setName)
            #
            # #Per ogni set genera labels/annotazioni
            # generateLabels(setId,setName)
    else:
        setId = set["id"]
        setName = set["name"]

        # singleSetRequest(...) scarica e organizza in cartelle il singolo set
        singleSetRequest(setId, setName)

        # Per ogni set genera labels/annotazioni
        generateLabels(setId, setName)


#Lista set disponibili
# setID  SetName

# base1 Base Set
# base2 Jungle
# basep Wizards Black Star Promos
# wp W Promotional
# base3 Fossil
# jumbo Jumbo cards
# base4 Base Set 2
# base5 Team Rocket
# gym1 Gym Heroes
# gym2 Gym Challenge
# neo1 Neo Genesis
# neo2 Neo Discovery
# si1 Southern Islands
# neo3 Neo Revelation
# neo4 Neo Destiny
# lc Legendary Collection
# sp Sample
# ecard1 Expedition Base Set
# bog Best of game
# ecard2 Aquapolis
# ecard3 Skyridge
# ex1 Ruby & Sapphire
# ex2 Sandstorm
# np Nintendo Black Star Promos
# ex3 Dragon
# ex4 Team Magma vs Team Aqua
# ex5 Hidden Legends
# ex5.5 Poké Card Creator Pack
# tk-ex-latio EX trainer Kit (Latios)
# tk-ex-latia EX trainer Kit (Latias)
# ex6 FireRed & LeafGreen
# pop1 POP Series 1
# ex7 Team Rocket Returns
# ex8 Deoxys
# ex9 Emerald
# pop2 POP Series 2
# exu Unseen Forces Unown Collection
# ex10 Unseen Forces
# ex11 Delta Species
# ex12 Legend Maker
# tk-ex-p EX trainer Kit 2 (Plusle)
# tk-ex-n EX trainer Kit 2 (Ninun)
# pop3 POP Series 3
# ex13 Holon Phantoms
# pop4 POP Series 4
# ex14 Crystal Guardians
# ex15 Dragon Frontiers
# ex16 Power Keepers
# pop5 POP Series 5
# dp1 Diamond & Pearl
# dpp DP Black Star Promos
# dp2 Mysterious Treasures
# pop6 POP Series 6
# tk-dp-m DP trainer Kit (Manaphy)
# tk-dp-l DP trainer Kit (Lucario)
# dp3 Secret Wonders
# dp4 Great Encounters
# pop7 POP Series 7
# dp5 Majestic Dawn
# dp6 Legends Awakened
# pop8 POP Series 8
# dp7 Stormfront
# pl1 Platinum
# pop9 POP Series 9
# pl2 Rising Rivals
# pl3 Supreme Victors
# pl4 Arceus
# ru1 Pokémon Rumble
# hgss1 HeartGold SoulSilver
# hgssp HGSS Black Star Promos
# tk-hs-r HS trainer Kit (Raichu)
# tk-hs-g HS trainer Kit (Gyarados)
# hgss2 Unleashed
# hgss3 Undaunted
# hgss4 Triumphant
# col1 Call of Legends
# bw1 Black & White
# bwp BW Black Star Promos
# 2011bw Macdonald's Collection 2011
# bw2 Emerging Powers
# tk-bw-e HS trainer Kit (Excadrill)
# tk-bw-z HS trainer Kit (Zoroark)
# bw3 Noble Victories
# bw4 Next Destinies
# bw5 Dark Explorers
# 2012bw Macdonald's Collection 2012
# bw6 Dragons Exalted
# dv1 Dragon Vault
# bw7 Boundaries Crossed
# bw8 Plasma Storm
# bw9 Plasma Freeze
# bw10 Plasma Blast
# xyp XY Black Star Promos
# rc Radiant Collection
# bw11 Legendary Treasures
# xy0 Kalos Starter Set
# xya Yello A Alternate
# xy1 XY
# tk-xy-n XY trainer Kit (Noivern)
# tk-xy-sy XY trainer Kit (Sylveon)
# xy2 Flashfire
# 2014xy Macdonald's Collection 2014
# xy3 Furious Fists
# tk-xy-w XY trainer Kit (Wigglytuff)
# tk-xy-b XY trainer Kit (Bisharp)
# xy4 Phantom Forces
# xy5 Primal Clash
# dc1 Double Crisis
# tk-xy-latio XY trainer Kit (Latios)
# tk-xy-latia XY trainer Kit (Latias)
# xy6 Roaring Skies
# xy7 Ancient Origins
# xy8 BREAKthrough
# 2015xy Macdonald's Collection 2015
# xy9 BREAKpoint
# g1 Generations
# tk-xy-su XY trainer Kit (Suicune)
# tk-xy-p XY trainer Kit (Pikachu Libre)
# xy10 Fates Collide
# xy11 Steam Siege
# 2016xy Macdonald's Collection 2016
# xy12 Evolutions
# smp SM Black Star Promos
# sm1 Sun & Moon
# tk-sm-l SM trainer Kit (Lycanroc)
# tk-sm-r SM trainer Kit (Alolan Raichu)
# sm2 Guardians Rising
# 2017sm Macdonald's Collection 2017
# sm3 Burning Shadows
# sm3.5 Shining Legends
# sm4 Crimson Invasion
# sm5 Ultra Prism
# sm6 Forbidden Light
# sm7 Celestial Storm
# sm7.5 Dragon Majesty
# 2018sm Macdonald's Collection 2018
# sm8 Lost Thunder
# sm9 Team Up
# det1 Detective Pikachu
# sm10 Unbroken Bonds
# sm11 Unified Minds
# sm115 Hidden Fates
# sma Yellow A Alternate
# 2019sm Macdonald's Collection 2019
# sm12 Cosmic Eclipse
# swshp SWSH Black Star Promos
# swsh1 Sword & Shield
# swsh2 Rebel Clash
# swsh3 Darkness Ablaze
# fut2020 Pokémon Futsal 2020
# swsh3.5 Champion's Path
# swsh4 Vivid Voltage
# 2021swsh Macdonald's Collection 2021
# swsh4.5 Shining Fates
# swsh5 Battle Styles
# swsh6 Chilling Reign
# swsh7 Evolving Skies
# cel25 Celebrations
# swsh8 Fusion Strike
# swsh9 Brilliant Stars
# swsh10 Astral Radiance
# swsh10.5 Pokémon GO
# swsh11 Lost Origin
# swsh12 Silver Tempest
# swsh12.5 Crown Zenith
# svp SVP Black Star Promos
# sv01 Scarlet & Violet
# sv02 Paldea Evolved
# sv03 Obsidian Flames
# sv03.5 151
# sv04 Paradox Rift
# sv04.5 Paldean Fates
# sv05 Temporal Forces
# sv06 Twilight Masquerade
# sv06.5 Shrouded Fable
# sv07 Stellar Crown
# A1 Genetic Apex
# P-A Promos-A
# sv08 Surging Sparks
# A1a Mythical Island
# sv08.5 Prismatic Evolutions
# A2 Space-Time Smackdown