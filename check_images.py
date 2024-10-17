import os
from PIL import Image, ImageOps
from tqdm import tqdm
import shutil

def process_image(image_path):
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Auto-orientation of pixel data (with EXIF-orientation stripping)
            img = ImageOps.exif_transpose(img)

            # Get the current dimensions
            width, height = img.size
            
            # Calculate padding to make it square
            pad_width = (max(width, height) - img.width) // 2
            pad_height = (max(width, height) - img.height) // 2

            # Determine the fill color based on the image mode
            if img.mode == "RGB":
                fill_color = (0, 0, 0)  # Black in RGB
            elif img.mode == "L":  # Grayscale
                fill_color = 0  # Black in grayscale
            else:
                fill_color = 0  # Default to black for unsupported modes

            # Pad to make it square
            img = ImageOps.expand(img, (pad_width, pad_height, pad_width, pad_height), fill=fill_color)

            # Resize to 224x224
            img = img.resize((224, 224), Image.LANCZOS)

            # Save the processed image
            img.save(image_path, optimize=True, quality=100)
    except Exception as e:
        tqdm.write(f"Error processing image: {image_path} - {e}")

def check_and_process_images(directory):
    correct_size = 0
    incorrect_size = 0
    incorrect_files = []

    # Get all PNG files in the directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]

    # Use tqdm for a progress bar
    for filename in tqdm(image_files, desc="Checking images"):
        file_path = os.path.join(directory, filename)
        try:
            with Image.open(file_path) as img:
                if img.size == (224, 224):
                    correct_size += 1
                else:
                    incorrect_size += 1
                    incorrect_files.append((filename, img.size))

                    # Process and resize the image if the size is incorrect
                    process_image(file_path)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            incorrect_size += 1
            incorrect_files.append((filename, "Error"))

    return correct_size, incorrect_size, incorrect_files

def is_image_corrupted(image_path):
    """
    Check if a given image is corrupted by trying to open it.
    Returns True if the image is corrupted, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # verify checks for corruption
        return False
    except (IOError, SyntaxError) as e:
        return True

def remove_corrupted_images(directory):
    """
    Find corrupted PNG images in the specified directory, remove them, and delete the directory if it becomes empty.
    """
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return 0, []

    corrupted_files = []
    print(f"Checking directory: {directory}")

    # List of PNG files in the directory
    png_files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]

    if not png_files:
        print(f"No PNG files in: {directory}")
    
    for file_name in png_files:
        file_path = os.path.join(directory, file_name)

        # Check if the image is corrupted
        if is_image_corrupted(file_path):
            print(f"Corrupted image found and removed: {file_path}")
            os.remove(file_path)
            corrupted_files.append(file_name)
        else:
            print(f"Image is fine: {file_path}")
    
    # Check if the directory is empty after removing images
    if not os.listdir(directory):
        print(f"Removing empty directory: {directory}")
        shutil.rmtree(directory)

    return len(corrupted_files), corrupted_files

if __name__ == "__main__":
    # Define the base directory to start checking for PNGs
    base_directory = "datasets/tcg_magic/data/train"

    # Check for corrupted images
    print(f"Starting corruption check in: {base_directory}")
    corrupted_count, corrupted_files = remove_corrupted_images(base_directory)

    # Check and process images
    correct, incorrect, wrong_files = check_and_process_images(base_directory)

    # Print results
    print(f"\nSummary of operations:")
    print(f"Corrupted images removed: {corrupted_count}")
    if corrupted_count > 0:
        print("Files removed due to corruption:")
        for file in corrupted_files:
            print(f" - {file}")

    print(f"Images with size 224x224: {correct}")
    print(f"Images processed to correct size: {incorrect}")

    if incorrect > 0:
        print("\nFiles with incorrect sizes before processing:")
        for file, size in wrong_files:
            print(f"{file}: {size}")
