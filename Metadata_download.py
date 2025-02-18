
import requests

from getReqFunctions import singleSetRequest
from labelsGenerator import generateLabels

#Recupera tutti i set disponibili
API_URL_ALLSETS= f"https://api.tcgdex.net/v2/en/sets"
response = requests.get(API_URL_ALLSETS)
sets=response.json()

#Cicla su ogni set disponibile
for set in sets:
    setId=set["id"]
    setName= set["name"]

    #singleSetRequest(...) scarica e organizza in cartelle il singolo set
    singleSetRequest(setId,setName)

    #Per ogni set genera labels/annotazioni
    generateLabels(setId,setName)



