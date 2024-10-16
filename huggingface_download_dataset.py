from datasets import load_dataset
import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load dataset from Hugging Face
ds = load_dataset("acidtib/tcg-magic", split="train")

# Create a directory to save the images
output_dir = 'datasets/tcg_magic/data/train'
os.makedirs(output_dir, exist_ok=True)

# Define a function to save a single image and update progress
def save_image(item, pbar):
    image_data = item['image']
    label = str(item['label']).replace("/", "_")
    image_path = os.path.join(output_dir, f'{label}.png')
    
    # Skip if the file already exists
    if image_data and not os.path.exists(image_path):
        image_data.save(image_path)
    
    # Update the progress bar after saving
    pbar.update(1)

# Initialize tqdm progress bar
with tqdm(total=len(ds), desc="Saving images") as pbar:
    # Create a ThreadPoolExecutor
    workers = os.cpu_count() if os.cpu_count() is not None else 2
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(save_image, item, pbar) for item in ds]
        # Wait for all threads to complete
        for future in futures:
            future.result()  # This will ensure we wait for the completion of all futures

print(f"Images saved to {output_dir}")
