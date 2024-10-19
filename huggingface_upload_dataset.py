import json
from datasets import load_dataset
from huggingface_hub import login

# Load the dataset from the local folder
dataset = load_dataset("imagefolder", 
                       data_dir="/media/acid/turtle/datasets/tcg_magic/data",
                       split={
                         "train": "train", 
                         "test": "test", 
                         "valid": "validation"
                        })

# Print the number of examples in each split
print(f"Number of examples in train: {len(dataset['train'])}")
print(f"Number of examples in test: {len(dataset['test'])}")
print(f"Number of examples in test: {len(dataset['valid'])}")

print("Uploading to Hugging Face Hub...")

# Push the dataset to Hugging Face hub
dataset['train'].push_to_hub("acidtib/tcg-magic-cards", split="train")
dataset['test'].push_to_hub("acidtib/tcg-magic-cards", split="test")
dataset['valid'].push_to_hub("acidtib/tcg-magic-cards", split="validation")

print("Upload complete!")