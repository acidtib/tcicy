import os
import requests
import numpy as np
import orjson
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img, load_img, img_to_array
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps
from tqdm import tqdm

# Define the maximum number of images to download from each JSON file
# Set to None if you want to download all cards
MAX_IMAGES_PER_FILE = 15

# Define the number of threads to use for downloading images
NUM_THREADS = os.cpu_count() if os.cpu_count() is not None else 2

# Resize and Pad Images
PROCESS_IMAGES = True

# Generate augmented images
GENERATE_AUGMENTED = True
AUGMENTED_AMOUNT = 7

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


# Utility function to ensure a directory exists
def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

# Utility function to clean the URL
def clean_url(url):
    return url.split('?')[0]

# Utility function to download a single JSON file
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

# Utility function to download images from a JSON file
def download_images_from_json(json_file, images_directory):
    with open(json_file, 'r') as file:
        data = orjson.loads(file.read())

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
        list(tqdm(executor.map(lambda url: download_image(clean_url(url), images_directory), image_urls), desc="Downloading Cards", unit="img", total=len(image_urls)))

# Utility function to download a single image
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

    # Check if the image file already exists and skip if it does
    if os.path.exists(image_path.replace(".png", ".jpg")):
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

# Utility function to process an image
def process_image(image_path):
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Auto-orientation of pixel data (with EXIF-orientation stripping)
            img = ImageOps.exif_transpose(img)
            
            # Create a new black background image
            new_size = (298, 298)
            new_img = Image.new("RGB", new_size, (0, 0, 0))
            
            # # Resize the original image while maintaining aspect ratio
            img.thumbnail((298, 298), Image.LANCZOS)
            
            # # Calculate position to paste (center)
            paste_position = ((new_size[0] - img.size[0]) // 2,
                              (new_size[1] - img.size[1]) // 2)
            
            # Paste the resized image onto the black background
            new_img.paste(img, paste_position)
            
            # Create the new file path with .jpg extension
            new_image_path = os.path.splitext(image_path)[0] + ".jpg"
            
            # Save the processed image
            new_img.save(new_image_path, format="JPEG", optimize=True, quality=95)
            
            # Remove the original image
            os.remove(image_path)
    except Exception as e:
        tqdm.write(f"Error processing image: {image_path} - {e}")

# Utility function to generate augmented images
def generate_augmented_images(img_path, save_dir, total_number=5):
    img_path = img_path.replace(".png", ".jpg")
    
    # Check if the directory already has the required augmented images
    existing_images = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')]) - 1
    
    if existing_images >= total_number:
        return False  # No need to generate new images, return False to indicate skipping

    # ImageDataGenerator configuration
    data_gen = ImageDataGenerator(
        rotation_range=10,          # Limit rotation to prevent excessive skewing
        width_shift_range=0.05,     # Slight shifts
        height_shift_range=0.05,    
        shear_range=0.0,            # Disable shearing to maintain aspect ratio
        zoom_range=[0.95, 1.05],    # Keep zoom minimal to prevent excessive distortion
        horizontal_flip=False,       # Keep this if cards are symmetrical
        fill_mode='constant',       # Keep the background constant when shifting/rotating
        cval=0                      # Set the background color (for shifted parts) to black or a suitable constant
    )

    # Load image and convert it to a tensor
    img = load_img(img_path, color_mode='rgb')
    arr = img_to_array(img)
    tensor_image = arr.reshape((1, ) + arr.shape)

    # Generate augmented images and save in order
    for i in range(total_number):
        # Generate a batch
        batch = next(data_gen.flow(x=tensor_image, batch_size=1))
        
        # Ensure the output retains the original size to prevent stretching
        batch[0] = np.clip(batch[0], 0, 255).astype(np.uint8)

        # Construct a filename with four leading zeros
        file_name = f"{save_dir}/{i+1:04d}.jpg"
        
        # Save the current image
        save_img(file_name, batch[0])  # Saving the generated image
    
    return True  # Images generated, return True to indicate success

# Utility function to copy up to 4 images to test directory
def copy_images_to_test(src, dst):
    # Initialize existing_images as an empty list
    existing_images = []
    
    ensure_directory_exists(dst)
    
    existing_images = [f for f in os.listdir(dst) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(existing_images) >= 4:
        return 0  # Skip if there are already 4 or more images
    
    # Get all image files in the source directory
    image_files = [f for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Select up to 4 images to copy, excluding any that already exist in dst
    images_to_copy = [img for img in image_files[:4] if img not in existing_images]

    # Copy the selected images
    for image in images_to_copy:
        shutil.copy2(os.path.join(src, image), os.path.join(dst, image))

    return len(images_to_copy)

# Utility function to set up test data
def setup_test_data(train_dir, test_dir):
    print("Setting up test data...")
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    
    # Get all subdirectories in the train directory
    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"Found {len(subdirs)} subdirectories in train directory")
    
    if not subdirs:
        print("No subdirectories found in train directory. Nothing to copy.")
        return
    
    # Use ThreadPoolExecutor to copy images in parallel
    with ThreadPoolExecutor() as executor:
        # Create a list to keep track of futures
        futures = []
        
        # Submit tasks to the executor for each subdirectory
        for dir_name in subdirs:
            src = os.path.join(train_dir, dir_name)
            dst = os.path.join(test_dir, dir_name)
            futures.append(executor.submit(copy_images_to_test, src, dst))
        
        # Process the futures as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying to test"):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error copying images: {e}")

    print(f"Copied all {len(subdirs)} directories to test set, copying up to 4 images in each.")

def copy_image_to_validation(src, dst):
    ensure_directory_exists(dst)
        
    existing_images = [f for f in os.listdir(dst) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(existing_images) >= 1:
        return 0  # Skip if there's already an image
    
    # Get all image files in the source directory
    image_files = [f for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Select 1 image to copy
    if image_files:
        image_to_copy = image_files[0]
        shutil.copy2(os.path.join(src, image_to_copy), os.path.join(dst, image_to_copy))
        return 1
    return 0

def setup_validation_data(train_dir, valid_dir):
    print("Setting up validation data...")
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {valid_dir}")
    
    # Get all subdirectories in the train directory
    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"Found {len(subdirs)} subdirectories in train directory")
    
    if not subdirs:
        print("No subdirectories found in train directory. Nothing to copy.")
        return
    
    # Use ThreadPoolExecutor to copy images in parallel
    with ThreadPoolExecutor() as executor:
        # Create a list to keep track of futures
        futures = []
        
        # Submit tasks to the executor for each subdirectory
        for dir_name in subdirs:
            src = os.path.join(train_dir, dir_name)
            dst = os.path.join(valid_dir, dir_name)
            futures.append(executor.submit(copy_image_to_validation, src, dst))
        
        # Process the futures as they complete
        copied_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying to validation"):
            try:
                copied_count += future.result()
            except Exception as e:
                print(f"Error copying image: {e}")

    print(f"Copied {copied_count} images from {len(subdirs)} directories to validation set.")
    
                
# Main function to download and process data
def main():
    print("\nFetching data from Scryfall...\n")
    directory = 'datasets/tcg_magic'
    ensure_directory_exists(directory)
    images_directory = os.path.join(directory, "data", "train")
    test_directory = os.path.join(directory, "data", "test")
    valid_directory = os.path.join(directory, "data", "valid")

    ensure_directory_exists(images_directory)
    ensure_directory_exists(test_directory)
    ensure_directory_exists(valid_directory)
    
    # Check if all required JSON files exist
    required_json_files = [f"{directory}/{type}.json" for type in BULK_DATA_TYPES]
    existing_json_files = [file for file in required_json_files if os.path.exists(file)]

    # If all files exist, skip the API call
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
    print("Loading JSON files...")
    for json_file in json_files:
        if json_file:  # If the download was successful
            with open(json_file, 'r') as file:
                data = orjson.loads(file.read())
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
    
    # Generate augmented images
    if GENERATE_AUGMENTED:
        print(f"Generating {AUGMENTED_AMOUNT} augmented images for each image in the train directory")
        
        # Get total number of images to process
        total_images = sum(1 for root, _, files in os.walk(images_directory) for file in files if file.endswith("0000.jpg"))
        
        # Create a tqdm progress bar for the entire process
        with tqdm(total=total_images, desc="Generating augmented images", unit="img") as pbar:
            with ThreadPoolExecutor() as executor:
                futures = []
                for root, _, files in os.walk(images_directory):
                    for file in files:
                        if file.endswith("0000.jpg"):
                            image_path = os.path.join(root, file)
                            # Submit the image processing task to the executor
                            futures.append(executor.submit(generate_augmented_images, image_path, os.path.dirname(image_path), AUGMENTED_AMOUNT))
                
                # Update the progress bar as each task completes
                for future in as_completed(futures):
                    pbar.update(1)

    # Setup test data
    setup_test_data(images_directory, test_directory)
    
    # Setup validation data
    setup_validation_data(images_directory, valid_directory)


if __name__ == "__main__":
    main()