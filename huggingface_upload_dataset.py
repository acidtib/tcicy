import json
from datasets import load_dataset
from huggingface_hub import login

# Load the dataset from the local folder
dataset = load_dataset("imagefolder", 
                       data_dir="./datasets/tcg_magic/data",
                       split={"train": "train", "test": "test"})

# Print the number of examples in each split
print(f"Number of examples in train: {len(dataset['train'])}")
print(f"Number of examples in test: {len(dataset['test'])}")

# Print the first few rows of the train dataset
print(json.dumps(dataset['train'][:3], indent=4))

print("Uploading to Hugging Face Hub...")

# Push the dataset to Hugging Face hub
dataset['train'].push_to_hub("acidtib/tcg-magic", split="train")
dataset['test'].push_to_hub("acidtib/tcg-magic", split="test")

print("Upload complete!")