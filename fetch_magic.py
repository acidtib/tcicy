import os
import requests
import json
import numpy as np
import random
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
from tqdm import tqdm

# Define the maximum number of images to download from each JSON file
# Set to None if you want to download all cards
MAX_IMAGES_PER_FILE = 420

# Define the number of threads to use for downloading images
NUM_THREADS = os.cpu_count() * 2 if os.cpu_count() is not None else 2

# Resize and Pad Images
PROCESS_IMAGES = True

# Generate augmented images
GENERATE_AUGMENTED = True
AUGMENTED_AMOUNT = 49

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

def setup_test_data(train_dir, test_dir):
    print("Setting up test data...")
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    
    # Ensure the test directory exists
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all subdirectories in the train directory
    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"Found {len(subdirs)} subdirectories in train directory")
    
    if not subdirs:
        print("No subdirectories found in train directory. Nothing to copy.")
        return
    
    # Copy all directories to test, but only 5 images from each
    for dir_name in tqdm(subdirs, desc="Copying to test"):
        src = os.path.join(train_dir, dir_name)
        dst = os.path.join(test_dir, dir_name)
        os.makedirs(dst, exist_ok=True)
        
        # Get all image files in the source directory
        image_files = [f for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Select up to 5 images to copy
        images_to_copy = image_files[:5] if len(image_files) > 5 else image_files
        
        # Copy the selected images
        for image in images_to_copy:
            shutil.copy2(os.path.join(src, image), os.path.join(dst, image))
    
    print(f"Copied all {len(subdirs)} directories to test set, copying up to 5 images in each.")
    
def generate_augmented_images(img_path, save_dir, total_number=10):
    # ImageDataGenerator configuration
    data_gen = ImageDataGenerator(
        rotation_range=10,          # Limit rotation to prevent excessive skewing
        width_shift_range=0.05,     # Slight shifts
        height_shift_range=0.05,    
        shear_range=0.0,            # Disable shearing to maintain aspect ratio
        zoom_range=[0.95, 1.05],    # Keep zoom minimal to prevent excessive distortion
        horizontal_flip=True,       # Keep this if cards are symmetrical
        fill_mode='constant',       # Keep the background constant when shifting/rotating
        cval=0                      # Set the background color (for shifted parts) to black or a suitable constant
    )

    # Load image and convert it to a tensor
    img = load_img(img_path, color_mode='rgb')
    arr = img_to_array(img)
    tensor_image = arr.reshape((1, ) + arr.shape)

    # Create save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate augmented images and save in order
    for i in range(total_number):
        # Generate a batch
        batch = next(data_gen.flow(x=tensor_image, batch_size=1))
        
        # Ensure the output retains the original size to prevent stretching
        batch[0] = np.clip(batch[0], 0, 255).astype(np.uint8)

        # Construct a filename with four leading zeros
        file_name = f"{save_dir}/{i+1:04d}.png"
        
        # Save the current image
        save_img(file_name, batch[0])  # Saving the generated image
        
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
    original_image_name = os.path.basename(image_url)
    # Replace - with _ in the image filename
    original_image_name = original_image_name.replace('-', '_')
    
    # Create a directory with the same name as the original image (without extension)
    image_dir_name = os.path.splitext(original_image_name)[0]
    image_dir_path = os.path.join(images_directory, image_dir_name)
    os.makedirs(image_dir_path, exist_ok=True)
    
    # Set the full path for the image file, now named 0000.png
    image_path = os.path.join(image_dir_path, '0000.png')

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
                
            if GENERATE_AUGMENTED:
                generate_augmented_images(image_path, image_dir_path, total_number=AUGMENTED_AMOUNT)

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
            
            # Create a new black background image
            new_size = (224, 224)
            new_img = Image.new("RGB", new_size, (0, 0, 0))
            
            # Resize the original image while maintaining aspect ratio
            img.thumbnail((224, 224), Image.LANCZOS)
            
            # Calculate position to paste (center)
            paste_position = ((new_size[0] - img.size[0]) // 2,
                              (new_size[1] - img.size[1]) // 2)
            
            # Paste the resized image onto the black background
            new_img.paste(img, paste_position)
            
            # Save the processed image
            new_img.save(image_path, optimize=True, quality=95)
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
    images_directory = f"{directory}/data/train"
    os.makedirs(images_directory, exist_ok=True)
    test_directory = f"{directory}/data/test"
    os.makedirs(test_directory, exist_ok=True)

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
            
    # Setup test data
    setup_test_data(images_directory, test_directory)

if __name__ == "__main__":
    main()