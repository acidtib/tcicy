import glob
import os
from datasets import Dataset, DatasetDict, Image, ClassLabel, Features

DATA_PATH = '/media/acid/turtle/datasets/tcg_magic'

OUTPUT_PATH = '/media/acid/turtle/huggingface/tcg-magic-cards/data'

# Function to get image paths
def get_image_paths(directory):
    return glob.glob(DATA_PATH + "/data/" + directory + "/**/*.jpg", recursive=True)

# Function to get labels
def get_labels(data_path):
    return [os.path.basename(os.path.dirname(path)) for path in data_path]

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
train_paths = get_image_paths("train")
test_paths = get_image_paths("test")
valid_paths = get_image_paths("valid")

# Create labels based on folder structure or filenames
print("Creating labels...")
train_labels = get_labels(train_paths)
test_labels = get_labels(test_paths)
valid_labels = get_labels(valid_paths)

# Create dataset for each split
print("Creating datasets...")
train_dataset = create_split_dataset(train_paths, train_labels)
test_dataset = create_split_dataset(test_paths, test_labels)
valid_dataset = create_split_dataset(valid_paths, valid_labels)

# Combine splits into a DatasetDict
print("Combining dataset...")
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "valid": valid_dataset
})

# save as Arrow locally
print("Saving dataset...")
dataset_dict.save_to_disk(
    OUTPUT_PATH,
    num_proc=os.cpu_count()
)

print("Done!")