import json
import os

import requests


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
        for set in sets_obtained:
            API_URL_SINGLE_SET= f"https://api.tcgdex.net/v2/en/sets/{set['id']}"
            response = requests.get(API_URL_SINGLE_SET)
            sets=response.json()


            for entry in series_list:
                if entry['id'] == sets['serie']['id']:
                    if sets.get('logo') is not None:
                        logo = sets['logo']
                    else:
                        logo = "---"
                    entry['sets'] = entry.get('sets', [])

                    entry['sets'].append({
                        'id': sets['id'],
                        'name': sets['name'],
                        'logo': logo

                    })
                    break

        generated_json_dir = "generated_json_dir"
        os.makedirs(generated_json_dir, exist_ok=True)
        collection_page_json="collection_page_json"
        subfolder_json = os.path.join(generated_json_dir, collection_page_json)
        os.makedirs(subfolder_json, exist_ok=True)
        file_path = os.path.join(subfolder_json, f"collections_logo_list.json")
        with open(file_path, "w") as json_file:
            json.dump(series_list, json_file, indent=4)

        print("Collections logo list saved to {}".format(file_path))
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
        print("Connection error")


if __name__ == "__main__":
    set_list_collections_generation()