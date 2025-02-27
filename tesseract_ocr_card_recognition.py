import os
import json
import cv2
import glob
import pytesseract
from fuzzywuzzy import fuzz

#FUNZIONA


# Constants optimized for mobile
IMG_SIZE = (192, 192)

class PokemonCardLocalRecognizer:
    def __init__(self, json_folder):
        self.json_folder = json_folder
        self.card_data = {}
        self.label_map_path = 'card_label_map.json'

    def load_card_data(self):
        """Load card data from raw JSON files"""
        json_files = glob.glob(os.path.join(self.json_folder, "*.json"))
        card_data = {}
        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if "name" not in data:
                        print(f"Warning: 'name' missing in {json_path}. Skipping.")
                        continue
                    card_name = data["name"]
                    set_name = data.get("set", {}).get("name", "Unknown")
                    card_data[f"{card_name} ({set_name})"] = {
                        "name": card_name,
                        "set": set_name,
                        "json_path": json_path
                    }
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in {json_path}. Skipping.")
            except Exception as e:
                print(f"Error processing {json_path}: {str(e)}. Skipping.")
        print(f"Loaded {len(card_data)} cards from raw JSONs")
        return card_data

    def preprocess_image_for_ocr(self, image):
        """Preprocess image for OCR, focusing on name area"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)  # Increase contrast
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Refine cropping for Base Set name area (top-left)
        height, width = thresh.shape
        name_area = thresh[int(height * 0.05):int(height * 0.15), int(width * 0.05):int(width * 0.5)]
        return name_area

    def extract_card_name(self, image):
        """Extract text from image using OCR and clean it"""
        preprocessed = self.preprocess_image_for_ocr(image)
        text = pytesseract.image_to_string(preprocessed, config='--psm 6')
        # Clean text: keep only letters and spaces, remove numbers and symbols
        cleaned_text = ''.join(c for c in text if c.isalpha() or c.isspace()).strip()
        return cleaned_text

    def match_card_name(self, extracted_text, top_k=5):
        """Match extracted text against known card names"""
        matches = []
        for card_key, card_info in self.card_data.items():
            if not isinstance(card_info, dict):
                print(f"Error: Invalid card_info for {card_key}: {card_info}")
                continue
            card_name = card_info["name"]
            score = fuzz.ratio(extracted_text.lower(), card_name.lower()) / 100.0
            matches.append({
                "card_name": card_name,
                "set": card_info["set"],
                "confidence": score
            })
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches[:top_k]

    def recognize_card(self, image):
        """Recognize card by extracting and matching its name"""
        extracted_text = self.extract_card_name(image)
        print(f"Extracted text: {extracted_text}")
        if not extracted_text:
            return {"matches": [{"card_name": "Unknown", "set": "Unknown", "confidence": 0.0}]}
        matches = self.match_card_name(extracted_text)
        return {"matches": matches}

    def save_model(self):
        """Save card data as a label map"""
        self.card_data = self.load_card_data()
        with open(self.label_map_path, 'w') as f:
            json.dump(self.card_data, f)
        print("Card data saved")

    def load_existing_model(self):
        """Load card data if it exists"""
        if os.path.exists(self.label_map_path):
            with open(self.label_map_path, 'r') as f:
                loaded_data = json.load(f)
            # Ensure loaded_data is a dictionary of dictionaries
            if isinstance(loaded_data, dict) and all(isinstance(v, dict) for v in loaded_data.values()):
                self.card_data = loaded_data
                print(f"Loaded existing card data with {len(self.card_data)} cards")
                return True
            else:
                print(f"Error: Invalid structure in {self.label_map_path}. Regenerating.")
                self.save_model()
                return True
        return False

class MobileApp:
    def __init__(self, model_files):
        self.recognizer = PokemonCardLocalRecognizer("")
        self.recognizer.label_map_path = model_files['labels']
        self.recognizer.load_existing_model()

    def recognize_card(self, camera_image):
        return self.recognizer.recognize_card(camera_image)

if __name__ == "__main__":
    json_folder = "annotations_dir/labels_BaseSet"
    label_map_path = 'card_label_map.json'

    # Delete existing card_label_map.json to force regeneration
    if os.path.exists(label_map_path):
        os.remove(label_map_path)
        print(f"Deleted existing {label_map_path} to force regeneration")

    recognizer = PokemonCardLocalRecognizer(json_folder)
    recognizer.label_map_path = label_map_path

    if not recognizer.load_existing_model():
        print("Generating card name list...")
        recognizer.save_model()

    mobile_app = MobileApp({'labels': label_map_path})

    test_image = cv2.imread("./test_images/hit.png")
    if test_image is not None:
        result = mobile_app.recognize_card(test_image)
        print(result)
    else:
        print("Failed to load test image. Please provide a valid image path.")