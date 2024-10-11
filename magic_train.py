import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set up the distributed strategy to use all available GPUs
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

## Data Preprocessing
batch_size = 32 # 64 or 128
epochs = 200
img_width = 224 # MobileNetV2 input size
img_height = 224

data_dir = pathlib.Path("datasets/tcg_magic").with_suffix('')

image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)

train_generator = image_dataset_from_directory(
    data_dir,
    validation_split=0.15,
    subset="training",
    seed=123,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)

validation_generator = image_dataset_from_directory(
    data_dir,
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_generator.class_names
print(f"Class names: {class_names}")

# Save class names to a text file
class_names_path = 'models/tcg_magic/class_names.txt'
os.makedirs(os.path.dirname(class_names_path), exist_ok=True)
with open(class_names_path, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# Display sample images
plt.figure(figsize=(10, 10))
for images, labels in train_generator.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.savefig('results/sample_images.png')

## Data Augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Show augmented images
plt.figure(figsize=(10, 10))
for images, _ in train_generator.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.savefig('results/data_augmentation.png')

# Define the model checkpoint path
best_model_path = 'models/tcg_magic/best_model.keras'

# Create and compile the model within the strategy scope
with strategy.scope():
    # Load MobileNetV2 Pretrained Model
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                             include_top=False,
                             weights='imagenet')

    # Freeze the base_model
    base_model.trainable = False

    # # when we have more data
    # base_model.trainable = True
    # # Optionally, freeze some initial layers
    # for layer in base_model.layers[:100]:
    #     layer.trainable = False

    # Add new classification head
    model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation='softmax')
    ])

    # If the best model exists, load weights from it
    if os.path.exists(best_model_path):
        print(f"Loading weights from the best model at {best_model_path}")
        model.load_weights(best_model_path)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

# Display model summary
model.summary()

# Set up early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min')

## Training the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[early_stopping, checkpoint]
)

## Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))  # Get the actual number of epochs

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('results/training_results.png')

## Save the final model
final_model_path = 'models/tcg_magic/final_model.keras'
model.save(final_model_path)

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model.
tflite_model_path = 'models/tcg_magic/model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("Model and TFLite model saved.")
