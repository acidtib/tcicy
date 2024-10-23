import glob
import os
from datasets import Dataset, DatasetDict, Image, ClassLabel, Features

DATA_PATH = '/media/acid/turtle/datasets/tcg_magic'

# Function to create a dataset for a split
def create_split_dataset(image_paths, labels):
    # Get unique labels
    unique_labels = list(set(labels))
    
    # Define features
    features = Features({
        "image": Image(),
        "label": ClassLabel(names=unique_labels)
    })
    
    # Create a dictionary for the dataset
    data_dict = {
        "image": image_paths,
        "label": labels
    }
    
    # Create the dataset
    return Dataset.from_dict(data_dict, features=features)

# Gather image paths from train, test, and valid directories
print("Gathering image paths...")
train_paths = glob.glob(DATA_PATH + "/data/train/**/*.jpg", recursive=True)
valid_paths = glob.glob(DATA_PATH + "/data/valid/**/*.jpg", recursive=True)

# Create labels based on folder structure or filenames
print("Creating labels...")
train_labels = [os.path.basename(os.path.dirname(path)) for path in train_paths]
valid_labels = [os.path.basename(os.path.dirname(path)) for path in valid_paths]

# Create dataset for each split
print("Creating datasets...")
train_dataset = create_split_dataset(train_paths, train_labels)
valid_dataset = create_split_dataset(valid_paths, valid_labels)

# Combine splits into a DatasetDict
print("Combining dataset...")
dataset_dict = DatasetDict({
    "train": train_dataset,
    "valid": valid_dataset
})

# save as Arrow locally
print("Saving dataset...")
dataset_dict.save_to_disk(
    "/media/acid/turtle/huggingface/tcg-magic-cards/data",
    num_proc=os.cpu_count()
)

print("Done!")