from datasets import load_dataset
import os
from PIL import Image
from tqdm.contrib.concurrent import thread_map
from concurrent.futures import ThreadPoolExecutor

# Load dataset from Hugging Face
ds = load_dataset("acidtib/tcg-magic", split="train")

# Create a directory to save the images
output_dir = 'datasets/tcg_magic/data/train'
os.makedirs(output_dir, exist_ok=True)

# Define a function to save a single image
def save_image(item):
    image_data = item['image']  # Adjust this to your specific image field name
    image_path = os.path.join(output_dir, f'{item["label"]}.png')  # Adjust the extension as needed
    if image_data and not os.path.exists(image_path):
        # Save the image
        image_data.save(image_path)

# Create a ThreadPoolExecutor
workers = os.cpu_count() if os.cpu_count() is not None else 2
with ThreadPoolExecutor(max_workers=workers) as executor:
    list(thread_map(save_image, ds, max_workers=workers, desc="Saving images"))

print(f"Images saved to {output_dir}")