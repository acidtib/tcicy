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
    Traverse through the directory and its subdirectories to find corrupted PNG images.
    Remove corrupted images and delete the folder if the image was the only file.
    """
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    # Add print to see if we're entering the function correctly
    print(f"Checking directory: {directory}")

    for root, dirs, files in os.walk(directory, topdown=False):
        # List of PNG files in the current directory
        png_files = [f for f in files if f.lower().endswith('.png')]

        if not png_files:
            print(f"No PNG files in: {root}")
        
        for file_name in png_files:
            file_path = os.path.join(root, file_name)

            # Check if the image is corrupted
            if is_image_corrupted(file_path):
                print(f"Corrupted image found and removed: {file_path}")
                os.remove(file_path)
            else:
                print(f"Image is fine: {file_path}")
        
        # Check if the folder is empty after removing images
        if not os.listdir(root):
            print(f"Removing empty folder: {root}")
            shutil.rmtree(root)

if __name__ == "__main__":
    # Define the base directory to start checking for PNGs
    base_directory = "datasets/tcg_magic"

    # Debugging: Ensure we get some output to indicate the script is running
    print(f"Starting corruption check in: {base_directory}")
    
    remove_corrupted_images(base_directory)
