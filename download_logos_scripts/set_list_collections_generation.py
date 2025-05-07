import json
import os
import requests
from tqdm import tqdm


#TODO:pulirlo dai set che non voglio far vedere
def set_list_collections_generation():
    try:
        """
        Creates json to display set logos and list in "collections.dart" page
        in the Flutter application
        
        Returns:
            Create the json to "generated_json_dir/collection_page_json/collections_logo_list.json"
        """

        #Call for series list
        API_URL_SERIES = "https://api.tcgdex.net/v2/en/series"
        response = requests.get(API_URL_SERIES)
        series_obtained = response.json()
        series_list=[]
        for series in series_obtained:
            series_list.append({"id":series["id"],"name":series["name"]})

        #Calls for set lists
        API_URL_ALL_SETS= f"https://api.tcgdex.net/v2/en/sets"
        response = requests.get(API_URL_ALL_SETS)
        sets_obtained=response.json()


        #Call for every single set
        for set in tqdm(sets_obtained,unit="sets",desc=f"Processing sets list"):
            API_URL_SINGLE_SET= f"https://api.tcgdex.net/v2/en/sets/{set['id']}"
            response = requests.get(API_URL_SINGLE_SET)
            sets=response.json()


            for entry in series_list:
                if entry['id'] == sets['serie']['id']:
                    if sets.get('logo') is not None:
                        logo = sets['logo']
                    else:
                        logo = "---"
                    if sets.get('symbol') is not None:
                        symbol = sets['symbol']
                    else:
                        symbol = "---"
                    entry['sets'] = entry.get('sets', [])

                    entry['sets'].append({
                        'id': sets['id'],
                        'name': sets['name'],
                        'logo': logo,
                        'symbol': symbol,
                        'cardCount': sets['cardCount']

                    })
                    break

        generated_json_dir = "set_logos_generated_json_dir"
        os.makedirs(generated_json_dir, exist_ok=True)
        # collection_page_json="collection_page_json"
        # subfolder_json = os.path.join(generated_json_dir, collection_page_json)
        os.makedirs(generated_json_dir, exist_ok=True)
        file_path = os.path.join(generated_json_dir, f"collections_logo_list.json")
        with open(file_path, "w") as json_file:
            json.dump(series_list, json_file, indent=4)

    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
        print("Connection error")


def download_sets_logo_images(json_path):

    image_logo_path = "../assets/set_logos"
    os.makedirs(image_logo_path, exist_ok=True)

    with open(json_path, "r") as file:
        data = json.load(file)

    try:
        for category in data:
            for set_data in tqdm(category["sets"], unit="sets", desc=f"Processing {category['name']} sets"):
                logo = set_data.get("logo", "")
                if logo and logo != "---":
                    API_URL_SINGLE_LOGO = logo + ".png"

                    try:
                        logo_img = requests.get(API_URL_SINGLE_LOGO).content
                        logo_file_name = set_data.get("id") + (set_data.get("name").replace(" ", ""))
                        logo_file_path = os.path.join(image_logo_path, logo_file_name + ".png")

                        with open(logo_file_path, "wb") as img_file:
                            img_file.write(logo_img)

                    except requests.exceptions.RequestException as e:
                        print(f"Failed to download {API_URL_SINGLE_LOGO}: {e}")

    except Exception as e:
        print(f"Cannot recover logo: {e}")


if __name__ == "__main__":
    set_list_collections_generation()
    json_path="set_logos_generated_json_dir/collections_logo_list.json"
    download_sets_logo_images(json_path)