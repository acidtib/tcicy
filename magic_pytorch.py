#!/usr/bin/env python
# coding: utf-8

# ## Install dependencies

# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install datasets')
get_ipython().system('pip install evaluate')
get_ipython().system('pip install accelerate')
get_ipython().system('pip install pillow')
get_ipython().system('pip install torchvision')
get_ipython().system('pip install scikit-learn')


# In[2]:


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datasets import load_dataset
import os
from PIL import Image


# ## Login to Hugging Face Hub

# In[3]:


from huggingface_hub import notebook_login

notebook_login()


# ### 1. Prepare the dataset

# In[4]:


def prepare_data(cards):
    train_data = cards['train']
    test_data = cards['test']
    
    # Extract images and labels
    train_images = train_data['image']
    train_labels = train_data['label']
    test_images = test_data['image']
    test_labels = test_data['label']
    
    # Encode labels
    le = LabelEncoder()
    train_labels_encoded = le.fit_transform(train_labels)
    test_labels_encoded = le.transform(test_labels)
    
    return train_images, train_labels_encoded, test_images, test_labels_encoded, le


# ### 2. Create a custom dataset class

# In[5]:


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

# In[6]:


def get_model(num_classes):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ### 4. Training loop

# In[7]:


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# # Main execution

# In[8]:


# Load the dataset
cards = load_dataset("acidtib/tcg-magic")


# In[9]:


# Prepare data
train_images, train_labels, test_images, test_labels, label_encoder = prepare_data(cards)


# In[23]:


# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[24]:


# Create datasets
train_dataset = MTGDataset(train_images, train_labels, transform=transform)
test_dataset = MTGDataset(test_images, test_labels, transform=transform)


# In[25]:


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[ ]:


# Initialize model
num_classes = len(label_encoder.classes_)
model = get_model(num_classes)


# In[27]:


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)


# In[ ]:


# Train the model
trained_model = train_model(model, train_loader, test_loader, criterion, optimizer)


# In[29]:


# Save the model
torch.save(trained_model.state_dict(), 'mtg_card_classifier.pth')


# In[ ]:


# Save the label encoder
import joblib
joblib.dump(label_encoder, 'label_encoder.joblib')

