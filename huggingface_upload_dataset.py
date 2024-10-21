from datasets import load_dataset

# Load the dataset from the local folder
dataset = load_dataset("imagefolder", 
                        data_dir="datasets/tcg_magic/data",
                        split={
                            "train": "train", 
                            "test": "test", 
                            "valid": "validation"
                        },
                        streaming=True
)

print("Uploading to Hugging Face Hub...")

# Push the dataset to Hugging Face hub
dataset['train'].push_to_hub("acidtib/tcg-magic-cards", split="train")
dataset['test'].push_to_hub("acidtib/tcg-magic-cards", split="test")
dataset['valid'].push_to_hub("acidtib/tcg-magic-cards", split="validation")

print("Upload complete!")