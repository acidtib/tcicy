import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datasets import load_dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import tensorrt
import joblib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from torchvision.models import ResNet50_Weights


# ### 1. Prepare the dataset
def encode_labels(all_labels):
    le = LabelEncoder()
    return le.fit_transform(all_labels), le

def prepare_data(cards):
    train_data = cards['train']
    test_data = cards['test']
    
    # Extract images and labels
    train_images = train_data['image']
    train_labels = train_data['label']
    test_images = test_data['image']
    test_labels = test_data['label']
    
    # Combine train and test labels for encoding
    all_labels = train_labels + test_labels  # Assuming labels are lists
    
    # Encode labels using multi-threading
    with ThreadPoolExecutor() as executor:
        future = executor.submit(encode_labels, all_labels)
        labels_encoded, le = future.result()
    
    # Split the encoded labels back into train and test sets
    n_train = len(train_labels)
    train_labels_encoded = labels_encoded[:n_train]
    test_labels_encoded = labels_encoded[n_train:]
    
    return train_images, train_labels_encoded, test_images, test_labels_encoded, le


# ### 2. Create a custom dataset class
class MTGDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert RGBA to RGB
        if image.mode == 'RGBA':
            image = Image.new("RGB", image.size, (255, 255, 255))
            image.paste(self.images[idx], mask=self.images[idx].split()[3])

        if self.transform:
            image = self.transform(image)

        return image, label


# ### 3. Set up model architecture
def get_model(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ### 4. Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)  # Wrap model with DataParallel
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

    return model


def show_images(dataset, label_encoder, num_images=5, images_per_row=2):
    rows = (num_images + images_per_row - 1) // images_per_row  # Calculate the number of rows needed
    plt.figure(figsize=(images_per_row * 4, rows * 4))
    
    # Randomly pick distinct indices from the dataset
    random_indices = random.sample(range(len(dataset)), num_images)
    
    for i, idx in enumerate(random_indices):
        image, label = dataset[idx]
        image = image.permute(1, 2, 0)  # Change dimensions to HWC for plotting
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Unnormalize
        image = image.clip(0, 1)  # Clip values to be between 0 and 1

        ax = plt.subplot(rows, images_per_row, i + 1)
        ax.imshow(image)
        ax.set_title(f"Label: {label_encoder.classes_[label]}", fontsize=10)  # Keep label size the same
        ax.axis('off')

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust these values as needed for spacing
    plt.tight_layout()
    # plt.show()
    # Save the image to a file
    plt.savefig("models/tcg_magic/examples.png", dpi=300, bbox_inches="tight")
    plt.close()


# # Main execution
# Load the dataset
print("Loading dataset...")
# cards = load_dataset("acidtib/tcg-magic")
cards = load_dataset("imagefolder", data_dir="datasets/tcg_magic/data")


# Prepare data
print("Preparing data...")
train_images, train_labels, test_images, test_labels, label_encoder = prepare_data(cards)

# Define transforms
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create datasets
train_dataset = MTGDataset(train_images, train_labels, transform=transform)
test_dataset = MTGDataset(test_images, test_labels, transform=transform)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Show some training images
show_images(train_dataset, label_encoder, num_images=4)


# Initialize model
num_classes = len(label_encoder.classes_)
model = get_model(num_classes)

print("Number of classes:", num_classes)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)


# Train the model
trained_model = train_model(
    model, 
    train_loader, 
    test_loader, 
    criterion, 
    optimizer,
    num_epochs=15
)


# Save the model
model_path = 'models/tcg_magic/classifier.pth'
torch.save(trained_model.state_dict(), model_path)

# Save the label encoder
joblib.dump(label_encoder, 'models/tcg_magic/labels.joblib')

# Save configuration
config = {
    "num_classes": num_classes,
    "label_encoder": label_encoder.classes_.tolist()
}

with open('models/tcg_magic/config.json', 'w') as f:
    json.dump(config, f)