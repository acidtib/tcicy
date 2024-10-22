import glob
import os
from datasets import Dataset, DatasetDict, Image, ClassLabel, Features

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
train_paths = glob.glob("datasets/tcg_magic/data/train/**/*.jpg", recursive=True)
test_paths = glob.glob("datasets/tcg_magic/data/test/**/*.jpg", recursive=True)
valid_paths = glob.glob("datasets/tcg_magic/data/valid/**/*.jpg", recursive=True)

# Create labels based on folder structure or filenames
train_labels = [os.path.basename(os.path.dirname(path)) for path in train_paths]
test_labels = [os.path.basename(os.path.dirname(path)) for path in test_paths]
valid_labels = [os.path.basename(os.path.dirname(path)) for path in valid_paths]

# Create datasets for each split
train_dataset = create_split_dataset(train_paths, train_labels)
test_dataset = create_split_dataset(test_paths, test_labels)
valid_dataset = create_split_dataset(valid_paths, valid_labels)

# Combine splits into a DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "valid": valid_dataset
})

# Push all splits to Hugging Face Hub in one go
print("Uploading dataset with all splits")
dataset_dict.push_to_hub("acidtib/tcg-magic-cards", create_pr=True)

print("Dataset with splits uploaded to Hugging Face Hub successfully!")
