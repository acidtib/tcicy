
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
from transformers import TrainingArguments, Trainer
from PIL import Image
import evaluate
import numpy as np
import joblib
import tensorrt
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def process_subset(data, label_encoder):
    images = data['image']
    labels = data['label']
    labels_encoded = label_encoder.transform(labels)
    return images, labels_encoded

def prepare_data(cards):
    train_data = cards['train']
    test_data = cards['test']
    
    # Initialize LabelEncoder
    le = LabelEncoder()
    all_labels = train_data['label'] + test_data['label']
    le.fit(all_labels)
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        train_future = executor.submit(partial(process_subset, train_data, le))
        test_future = executor.submit(partial(process_subset, test_data, le))
        
        # Wait for both tasks to complete
        train_images, train_labels_encoded = train_future.result()
        test_images, test_labels_encoded = test_future.result()
    
    return train_images, train_labels_encoded, test_images, test_labels_encoded, le


# ### Custom dataset class
class MTGDataset(torch.utils.data.Dataset):
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

        return {"pixel_values": image, "label": label}


# ### Model architecture
class MTGClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        logits = self.resnet(pixel_values)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return (loss, logits) if loss is not None else logits


# ### Data collator
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# ### Compute metrics function for evaluation
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def show_images(dataset, label_encoder, num_images=5, images_per_row=2):
    rows = (num_images + images_per_row - 1) // images_per_row  # Calculate the number of rows needed
    plt.figure(figsize=(images_per_row * 4, rows * 4))
    
    # Randomly pick distinct indices from the dataset
    random_indices = random.sample(range(len(dataset)), num_images)
    
    for i, idx in enumerate(random_indices):
        sample = dataset[idx]
        image = sample["pixel_values"]
        label = sample["label"]
        
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
    plt.savefig("examples.png", dpi=300, bbox_inches="tight")
    plt.close()


# # Main execution
def main():
    # Load the dataset
    # cards = load_dataset("acidtib/tcg-magic")
    print("Loading dataset...")
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
    print("Number of training images:", len(train_dataset))
    print("Number of test images:", len(test_dataset))
    
    # Show some training images
    show_images(train_dataset, label_encoder, num_images=4)
    
    # Initialize model
    num_labels = len(label_encoder.classes_)
    model = MTGClassifier(num_labels)

    print("Number of labels:", num_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,  # Reduced from 16
        per_device_eval_batch_size=32,  # Reduced from 64
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # gradient_accumulation_steps=2,  # accumulate gradients over 2 steps
        # fp16=True,  # enable mixed precision training
        # dataloader_num_workers=2,  # use multiple workers for data loading
        # remove_unused_columns=False,  # keep all columns
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("tcg_classifier")

    # Save the label encoder
    joblib.dump(label_encoder, 'label_encoder.joblib')

    print("Training completed and model saved!")

if __name__ == "__main__":
    main()