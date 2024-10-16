import os
from PIL import Image
import shutil

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
        return

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
        else:
            print(f"Image is fine: {file_path}")
    
    # Check if the directory is empty after removing images
    if not os.listdir(directory):
        print(f"Removing empty directory: {directory}")
        shutil.rmtree(directory)

if __name__ == "__main__":
    # Define the base directory to start checking for PNGs
    base_directory = "datasets/tcg_magic/data/train"

    # Debugging: Ensure we get some output to indicate the script is running
    print(f"Starting corruption check in: {base_directory}")
    
    remove_corrupted_images(base_directory)