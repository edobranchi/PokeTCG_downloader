# Pokémon TCG Set Fetcher

## Overview
This script fetches Pokémon Trading Card Game (TCG) set metadata, downloads card images, and generates annotations for each card. It interacts with the TCGdex API to retrieve up-to-date information on card sets and individual cards.

## Features
- Fetch metadata for all available Pokémon TCG sets.
- Download card images in a specified format and quality.
- Generate labels/annotations for each card, storing them in JSON format.

## Requirements
- Python 3.x
- `requests` library
- `tqdm` library
- Internet connection

Install dependencies with:
```sh
pip install requests tqdm
```

## Configuration
Modify the following global variables to adjust image settings:
```python
IMAGE_QUALITY = "low"  # Set image quality (e.g., "low", "high")
IMG_EXTENSION = "png"   # Set image file extension (e.g., "png", "jpg")
```

## Usage
### Run the script
```sh
python script.py
```

By default, it fetches metadata for all available sets, downloads card images, and generates labels. To fetch a specific set, modify:
```python
setId_to_download = "<SET_ID>"
setName_to_download = "<SET_NAME>"
```

### Functions
#### `singleSetRequest(setId, setName, metadata_dir="./metadata_dir")`
Fetches metadata for a given set and stores it in `metadata_dir`.

#### `ImgDownload(setId, setName, metadata_dir)`
Downloads images of all cards in a set and stores them in `assets/card_images`.

#### `generateLabels(setId, setName, metadata_dir="./metadata_dir")`
Generates JSON labels for each card, linking images and metadata, and saves them in `assets/image_labels`.

## Data Storage
- **Metadata files**: Stored in `./metadata_dir/cards_metadata_<SetName>.json`
- **Card images**: Stored in `./assets/card_images/`
- **Labels**: Stored in `./assets/image_labels/`

## Set IDs and set name
Here are the sets that can be fetched:
```
setID  SetName

base1 - Base Set
base2 - Jungle
basep - Wizards Black Star Promos
wp - W Promotional
base3 - Fossil
jumbo - Jumbo cards
base4 - Base Set 2
base5 - Team Rocket
gym1 - Gym Heroes
gym2 - Gym Challenge
neo1 - Neo Genesis
neo2 - Neo Discovery
si1 - Southern Islands
neo3 - Neo Revelation
neo4 - Neo Destiny
lc - Legendary Collection
sp - Sample
ecard1 - Expedition Base Set
bog - Best of game
ecard2 - Aquapolis
ecard3 - Skyridge
ex1 - Ruby & Sapphire
ex2 - Sandstorm
np - Nintendo Black Star Promos
ex3 - Dragon
ex4 - Team Magma vs Team Aqua
ex5 - Hidden Legends
ex5.5 - Poké Card Creator Pack
tk-ex-latio - EX trainer Kit (Latios)
tk-ex-latia - EX trainer Kit (Latias)
ex6 - FireRed & LeafGreen
pop1 - POP Series 1
ex7 - Team Rocket Returns
ex8 - Deoxys
ex9 - Emerald
pop2 - POP Series 2
exu - Unseen Forces Unown Collection
ex10 - Unseen Forces
ex11 - Delta Species
ex12 - Legend Maker
tk-ex-p - EX trainer Kit 2 (Plusle)
tk-ex-n - EX trainer Kit 2 (Ninun)
pop3 - POP Series 3
ex13 - Holon Phantoms
pop4 - POP Series 4
ex14 - Crystal Guardians
ex15 - Dragon Frontiers
ex16 - Power Keepers
pop5 - POP Series 5
dp1 - Diamond & Pearl
dpp - DP Black Star Promos
dp2 - Mysterious Treasures
pop6 - POP Series 6
tk-dp-m - DP trainer Kit (Manaphy)
tk-dp-l - DP trainer Kit (Lucario)
dp3 - Secret Wonders
dp4 - Great Encounters
pop7 - POP Series 7
dp5 - Majestic Dawn
dp6 - Legends Awakened
pop8 - POP Series 8
dp7 - Stormfront
pl1 - Platinum
pop9 - POP Series 9
pl2 - Rising Rivals
pl3 - Supreme Victors
pl4 - Arceus
ru1 - Pokémon Rumble
hgss1 - HeartGold SoulSilver
hgssp - HGSS Black Star Promos
tk-hs-r - HS trainer Kit (Raichu)
tk-hs-g - HS trainer Kit (Gyarados)
hgss2 - Unleashed
hgss3 - Undaunted
hgss4 - Triumphant
col1 - Call of Legends
bw1 - Black & White
bwp - BW Black Star Promos
2011bw - Macdonald's Collection 2011
bw2 - Emerging Powers
tk-bw-e - HS trainer Kit (Excadrill)
tk-bw-z - HS trainer Kit (Zoroark)
bw3 - Noble Victories
bw4 - Next Destinies
bw5 - Dark Explorers
2012bw - Macdonald's Collection 2012
bw6 - Dragons Exalted
dv1 - Dragon Vault
bw7 - Boundaries Crossed
bw8 - Plasma Storm
bw9 - Plasma Freeze
bw10 - Plasma Blast
xyp - XY Black Star Promos
rc - Radiant Collection
bw11 - Legendary Treasures
xy0 - Kalos Starter Set
xya - Yello A Alternate
xy1 - XY
tk-xy-n - XY trainer Kit (Noivern)
tk-xy-sy - XY trainer Kit (Sylveon)
xy2 - Flashfire
2014xy - Macdonald's Collection 2014
xy3 - Furious Fists
tk-xy-w - XY trainer Kit (Wigglytuff)
tk-xy-b - XY trainer Kit (Bisharp)
xy4 - Phantom Forces
xy5 - Primal Clash
dc1 - Double Crisis
tk-xy-latio - XY trainer Kit (Latios)
tk-xy-latia - XY trainer Kit (Latias)
xy6 - Roaring Skies
xy7 - Ancient Origins
xy8 - BREAKthrough
2015xy - Macdonald's Collection 2015
xy9 - BREAKpoint
g1 - Generations
tk-xy-su - XY trainer Kit (Suicune)
tk-xy-p - XY trainer Kit (Pikachu Libre)
xy10 - Fates Collide
xy11 - Steam Siege
2016xy - Macdonald's Collection 2016
xy12 - Evolutions
smp - SM Black Star Promos
sm1 - Sun & Moon
tk-sm-l - SM trainer Kit (Lycanroc)
tk-sm-r - SM trainer Kit (Alolan Raichu)
sm2 - Guardians Rising
2017sm - Macdonald's Collection 2017
sm3 - Burning Shadows
sm3.5 - Shining Legends
sm4 - Crimson Invasion
sm5 - Ultra Prism
sm6 - Forbidden Light
sm7 - Celestial Storm
sm7.5 - Dragon Majesty
2018sm - Macdonald's Collection 2018
sm8 - Lost Thunder
sm9 - Team Up
det1 - Detective Pikachu
sm10 - Unbroken Bonds
sm11 - Unified Minds
sm115 - Hidden Fates
sma - Yellow A Alternate
2019sm - Macdonald's Collection 2019
sm12 - Cosmic Eclipse
swshp - SWSH Black Star Promos
swsh1 - Sword & Shield
swsh2 - Rebel Clash
swsh3 - Darkness Ablaze
fut2020 - Pokémon Futsal 2020
swsh3.5 - Champion's Path
swsh4 - Vivid Voltage
2021swsh - Macdonald's Collection 2021
swsh4.5 - Shining Fates
swsh5 - Battle Styles
swsh6 - Chilling Reign
swsh7 - Evolving Skies
cel25 - Celebrations
swsh8 - Fusion Strike
swsh9 - Brilliant Stars
swsh10 - Astral Radiance
swsh10.5 - Pokémon GO
swsh11 - Lost Origin
swsh12 - Silver Tempest
swsh12.5 - Crown Zenith
svp - SVP Black Star Promos
sv01 - Scarlet & Violet
sv02 - Paldea Evolved
sv03 - Obsidian Flames
sv03.5 - 151
sv04 - Paradox Rift
sv04.5 - Paldean Fates
sv05 - Temporal Forces
sv06 - Twilight Masquerade
sv06.5 - Shrouded Fable
sv07 - Stellar Crown
A1 - Genetic Apex
P-A - Promos-A
sv08 - Surging Sparks
A1a - Mythical Island
sv08.5 - Prismatic Evolutions
A2 - Space-Time Smackdown
```
For a complete list, check the TCGdex API.

## Error Handling
- If a request fails, the script logs the error and moves to the next card/set.
- Skips downloading images that already exist in the destination folder.



