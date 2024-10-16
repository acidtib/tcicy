from datasets import load_dataset
from huggingface_hub import login

# Login to Hugging Face
# login(token="your_hugging_face_token")

# Load the dataset from the local folder
dataset = load_dataset("imagefolder", data_dir="./datasets/tcg_magic/data/train", split="train")

# Print the number of examples in the dataset
print(f"Number of examples in the dataset: {len(dataset)}")

# Print the first few rows of the dataset
print(dataset[:5])

print("Uploading to Hugging Face Hub...")

# Push the dataset to Hugging Face hub
dataset.push_to_hub("acidtib/tcg-magic")
