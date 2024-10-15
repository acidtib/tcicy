import os
import requests
import json

from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
from tqdm import tqdm

# Define the maximum number of images to download from each JSON file
# Set to None if you want to download all cards
MAX_IMAGES_PER_FILE = None

# Define the number of threads to use for downloading images
NUM_THREADS = os.cpu_count() * 2 if os.cpu_count() is not None else 2

# Resize and Pad Images
PROCESS_IMAGES = True

# Define the types of bulk data to download
BULK_DATA_TYPES = [
    # 203 MB
    # "unique_artwork", 

    # 146 MB
    # "oracle_cards",

    # 447 MB
    "default_cards",

    # 2.1 GB
    # "all_cards"
]

def download_json_data(type, download_uri, directory):
    filename = f"{directory}/{type}.json"
    
    if os.path.exists(filename):
        print(f"Skipping {type}, file already exists at {filename}")
        return
    
    print(f"Downloading {type} from {download_uri}...")
    response = requests.get(download_uri)
    
    if response.ok:
        with open(filename, 'wb') as file:
            file.write(response.content)
        
        print(f"Downloaded {type} to {filename}")
        return filename  # Return the filename for further processing
    else:
        print(f"Failed to download {type}: {response.status_code} {response.text}")
        return None

def download_image(image_url, images_directory):
    # Extract the image filename from the URL
    image_name = os.path.basename(image_url)
    # Replace - with _ in the image filename
    image_name = image_name.replace('-', '_')
    image_path = os.path.join(images_directory, image_name)

    # Check if the image file already exists
    if os.path.exists(image_path):
        return image_path

    try:
        response = requests.get(image_url)
        if response.ok:
            with open(image_path, 'wb') as img_file:
                img_file.write(response.content)

            if PROCESS_IMAGES:
                process_image(image_path)

            return image_path
        else:
            tqdm.write(f"Failed to download image: {image_url} - {response.status_code}")
            return None
    except Exception as e:
        tqdm.write(f"Error downloading image: {image_url} - {e}")
        return None

def download_images_from_json(json_file, images_directory):
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Extract the card images
    image_urls = [card['image_uris']['png'] for card in data if 'image_uris' in card and 'png' in card['image_uris']]
    card_ids = [card['id'] for card in data if 'image_uris' in card and 'png' in card['image_uris']]
    card_names = [card['name'] for card in data if 'image_uris' in card and 'png' in card['image_uris']]
    
    # Limit to MAX_IMAGES_PER_FILE if specified
    if MAX_IMAGES_PER_FILE is not None:
        image_urls = image_urls[:MAX_IMAGES_PER_FILE]
        card_ids = card_ids[:MAX_IMAGES_PER_FILE]
        card_names = card_names[:MAX_IMAGES_PER_FILE]

    # Download images concurrently with progress tracking
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Use tqdm to create a progress bar
        for image_path, card_id, card_name in zip(tqdm(executor.map(lambda url: download_image(clean_url(url), images_directory), image_urls), total=len(image_urls)), card_ids, card_names):
            if image_path:
                tqdm.write(f"Processed: {image_path}")

def clean_url(url):
    return url.split('?')[0]

def process_image(image_path):
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Auto-orientation of pixel data (with EXIF-orientation stripping)
            img = ImageOps.exif_transpose(img)

            # Get the current dimensions
            width, height = img.size
            
            # Calculate padding to make the image square
            pad_width = (max(width, height) - img.width) // 2
            pad_height = (max(width, height) - img.height) // 2
            
            # Pad to make it square
            img = ImageOps.expand(img, (pad_width, pad_height, pad_width, pad_height), fill=(0, 0, 0))  # Black padding

            # Resize to 224x224
            img = img.resize((224, 224), Image.LANCZOS)

            # Save the processed image
            img.save(image_path, optimize=True, quality=100)
    except Exception as e:
        tqdm.write(f"Error processing image: {image_path} - {e}")
            
def main():
    print("\n")
    print(" _       _            ")
    print("| |_ ___(_) ___ _   _ ")
    print("| __/ __| |/ __| | | |")
    print("| || (__| | (__| |_| |")
    print(" \\__\\___|_|\\___|\\__, |")
    print("                |___/ ")
    print("   tcicy  \n\n")
    print(f"Fetching {', '.join(BULK_DATA_TYPES)} from Scryfall")
    
    # Create the directory if it doesn't exist
    directory = 'datasets/tcg_magic'
    os.makedirs(directory, exist_ok=True)
    images_directory = f"{directory}/training"
    os.makedirs(images_directory, exist_ok=True)

    # Check if all required JSON files exist
    required_json_files = [f"{directory}/{type}.json" for type in BULK_DATA_TYPES]
    existing_json_files = [file for file in required_json_files if os.path.exists(file)]

    if len(existing_json_files) == len(required_json_files):
        print("All required JSON files already exist, skipping API call.")
        json_files = existing_json_files
    else:
        # Fetch all bulk data items
        response = requests.get('https://api.scryfall.com/bulk-data')
        data = response.json()

        # Check for successful response
        if data['object'] == 'list':
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                json_files = list(executor.map(
                    lambda item: download_json_data(item['type'], item['download_uri'], directory),
                    (item for item in data['data'] if item['type'] in BULK_DATA_TYPES)
                ))
        else:
            print("Failed to fetch bulk data.")
            return
        
    # Load the JSON data to get the total number of images
    total_images = 0
    for json_file in json_files:
        if json_file:  # If the download was successful
            with open(json_file, 'r') as file:
                data = json.load(file)
                total_images += len([card for card in data if 'image_uris' in card and 'png' in card['image_uris']])

    if MAX_IMAGES_PER_FILE is None:
        print(f"Downloading all {total_images} images")
    else:
        print(f"Downloading up to {MAX_IMAGES_PER_FILE} images per JSON file")

    print(f"Using {NUM_THREADS} concurrent threads")

    # Download images from each downloaded JSON file
    for json_file in json_files:
        if json_file:  # If the download was successful
            download_images_from_json(json_file, images_directory)

if __name__ == "__main__":
    main()