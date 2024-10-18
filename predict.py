from transformers import pipeline
from PIL import Image

# Load the image
image_path = "datasets/tcg_magic/data/valid/00a28b9d_1bd2_4ca4_a1e1_4138891d6739/0002.png"
image = Image.open(image_path)

pipe = pipeline("image-classification", model="acidtib/tcg-magic-cards")

# Classify the image
results = pipe(image)

# Print the results
print("Label                                 | Score")
print("-" * 53)
for result in results:
    print(f"{result['label']} | {result['score']:.4f}")