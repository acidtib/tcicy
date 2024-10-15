import tensorflow as tf
from tensorflow.keras import layers  # Import layers here
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Added import
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt  # Importing matplotlib for image visualization

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, image_size, shuffle=True, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
        self.classes = sorted(list(set([os.path.splitext(f)[0] for f in self.image_files])))  # Use full filename without extension
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.on_epoch_end()
        
        # Initialize the image data generator for augmentation
        self.datagen = None  # Initialize datagen to None
        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=10,  # Reduced rotation range
                width_shift_range=0.05,  # Reduced width shift
                height_shift_range=0.05,  # Reduced height shift
                shear_range=0.1,  # Reduced shear range
                zoom_range=0.1,  # Reduced zoom range
                horizontal_flip=False,  # Keep horizontal flip off to preserve orientation
                fill_mode='nearest'
            )

    def __len__(self):
        return int(np.ceil(len(self.image_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_files = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([self.load_image(os.path.join(self.directory, f)) for f in batch_files])
        batch_y = np.array([self.class_to_index[os.path.splitext(f)[0]] for f in batch_files])
        
        # Augment images if specified
        if self.augment:
            batch_x = self.augment_images(batch_x)

        return batch_x, tf.keras.utils.to_categorical(batch_y, num_classes=len(self.classes))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_files)

    def load_image(self, path):
        img = tf.keras.preprocessing.image.load_img(path)
        img = tf.keras.preprocessing.image.img_to_array(img)

        # Center the card in the image
        target_height, target_width = self.image_size
        height, width, _ = img.shape

        # Calculate the padding
        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)

        # Pad the image
        img = np.pad(img, ((pad_height // 2, pad_height - pad_height // 2), 
                            (pad_width // 2, pad_width - pad_width // 2), 
                            (0, 0)), mode='constant', constant_values=255)  # Use white padding

        # Resize to target size
        img = tf.image.resize(img, self.image_size)

        img = img / 255.0  # Normalize to [0,1]
        return img

    def augment_images(self, images):
        augmented_images = []
        for img in images:
            img = img.reshape((1,) + img.shape)  # Reshape for the generator
            # Generate augmented images
            for augmented_img in self.datagen.flow(img, batch_size=1):
                # Center the augmented image in the output array
                target_height, target_width = self.image_size
                augmented_img = augmented_img[0]  # Get the augmented image
                aug_height, aug_width, _ = augmented_img.shape
                
                # Calculate padding to center the card in the image
                pad_height = max(0, target_height - aug_height)
                pad_width = max(0, target_width - aug_width)
                augmented_img = np.pad(augmented_img, 
                                        ((pad_height // 2, pad_height - pad_height // 2), 
                                         (pad_width // 2, pad_width - pad_width // 2), 
                                         (0, 0)), 
                                        mode='constant', constant_values=255)  # Use white padding
                
                augmented_img = tf.image.resize(augmented_img, self.image_size)  # Resize to target size
                augmented_images.append(augmented_img)  # Add the centered augmented image
                break  # Stop after generating one augmented image
        return np.array(augmented_images)

    def save_sample_augmented_images(self, output_dir, n_samples=5):
        """ Save a few augmented sample images for inspection. """
        os.makedirs(output_dir, exist_ok=True)
        for idx, img_file in enumerate(self.image_files[:n_samples]):
            img = self.load_image(os.path.join(self.directory, img_file))
            img = img.numpy()  # Convert EagerTensor to NumPy array
            img = img.reshape((1,) + img.shape)  # Reshape for the generator
            
            for augmented_img in self.datagen.flow(img, batch_size=1):
                augmented_img = (augmented_img[0] * 255).astype(np.uint8)  # Convert back to uint8
                augmented_img_path = os.path.join(output_dir, f"augmented_{idx}.png")
                tf.keras.preprocessing.image.save_img(augmented_img_path, augmented_img)
                break  # Stop after saving one augmented image

# Save image visualizations
def save_image_visualization(images, labels, class_names, output_path):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])  # Removed .numpy() since images is already a NumPy array
        plt.title(class_names[labels[i].argmax()])  # Display class name
        plt.axis("off")
    plt.savefig(output_path)  # Save the figure
    plt.close()  # Close the figure to free memory

# Save augmented image visualizations
def save_augmented_visualization(images, generator, output_path):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        augmented_images = generator.augment_images(images[i:i + 1])  # Augment the first image
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0])  # Removed .numpy()
        plt.axis("off")
    plt.savefig(output_path)  # Save the figure
    plt.close()  # Close the figure to free memory

# Set up paths and parameters
data_dir = 'datasets/tcg_magic/training'
output_augmented_dir = 'datasets/tcg_magic/augmented_samples'  # Directory to save augmented images
image_size = (224, 224)
batch_size = 32

# Create custom data generators with augmentation enabled
train_generator = CustomDataGenerator(data_dir, batch_size, image_size, augment=True)
validation_generator = CustomDataGenerator(data_dir, batch_size, image_size)

# Save a few augmented samples for inspection
train_generator.save_sample_augmented_images(output_augmented_dir, n_samples=5)

# Save visualizations earlier in the code
image_visualization_path = 'visualizations/original_images.png'
augmented_visualization_path = 'visualizations/augmented_images.png'
os.makedirs('visualizations', exist_ok=True)  # Create directory if it doesn't exist

# Get the first batch of images and labels
images, labels = train_generator.__getitem__(0)

# Save original images visualization
save_image_visualization(images, labels, train_generator.classes, image_visualization_path)

# Save augmented images visualization
save_augmented_visualization(images, train_generator, augmented_visualization_path)

# Split data into train and validation
num_samples = len(train_generator.image_files)
num_train = int(0.8 * num_samples)
train_generator.image_files = train_generator.image_files[:num_train]
validation_generator.image_files = validation_generator.image_files[num_train:]

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(len(train_generator.classes), activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Check if a previously saved best model exists
best_model_path = 'models/tcg_magic/best_model.keras'
if os.path.exists(best_model_path):
    print("Loading existing best model...")
    model = tf.keras.models.load_model(best_model_path)
else:
    # Freeze the base model layers for initial training
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up TensorBoard
log_dir = "models/tcg_magic/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True),
    ModelCheckpoint('models/tcg_magic/best_model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.1, patience=3),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# Train or continue training the model
epochs = 80  # Increased epochs, EarlyStopping will prevent overfitting
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=callbacks
)

# Save the model
model.save(best_model_path)

# Fine-tuning
# Unfreeze some layers of the base model
print("Preparing to fine-tune...")
for layer in base_model.layers[-40:]:
    layer.trainable = True
    
# Recompile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training (fine-tuning)
print("Starting fine-tuning...")
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=80,
    callbacks=callbacks
)

print("Fine-tuning completed.")

# Save the fine-tuned model
model.save("models/tcg_magic/classifier_fine_tuned.keras")
print("Fine-tuned model saved.")
