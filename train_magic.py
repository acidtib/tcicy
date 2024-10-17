import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, shuffle=True, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
        self.classes = sorted(list(set([os.path.splitext(f)[0] for f in self.image_files])))
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.on_epoch_end()
        
        self.datagen = None
        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=False,
                fill_mode='constant',
                cval=0
            )

    def __len__(self):
        return int(np.ceil(len(self.image_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_files = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([self.load_image(os.path.join(self.directory, f)) for f in batch_files])
        batch_y = np.array([self.class_to_index[os.path.splitext(f)[0]] for f in batch_files])
        
        if self.augment:
            batch_x = self.augment_images(batch_x)

        return batch_x, tf.keras.utils.to_categorical(batch_y, num_classes=len(self.classes))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_files)

    def load_image(self, path):
        img = tf.keras.preprocessing.image.load_img(path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        # Ensure the image is in the correct format (224, 224, 3)
        if img.shape != (224, 224, 3):
            raise ValueError(f"Unexpected image shape: {img.shape}. Expected (224, 224, 3)")
        
        # Normalize the image
        img = img / 255.0
        return img

    def augment_images(self, images):
        augmented_images = []
        for img in images:
            img = img.reshape((1,) + img.shape)
            for augmented_img in self.datagen.flow(img, batch_size=1):
                augmented_img = augmented_img[0]
                
                # Preserve the original black padding
                mask = (img[0] != 0).any(axis=2)
                augmented_img[~mask] = 0
                
                augmented_images.append(augmented_img)
                break
        return np.array(augmented_images)

def save_image_visualization(images, labels, class_names, output_path):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i].argmax()])
        plt.axis("off")
    plt.savefig(output_path)
    plt.close()

def save_augmented_visualization(images, generator, output_path):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        augmented_images = generator.augment_images(images[i:i+1])
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0])
        plt.axis("off")
    plt.savefig(output_path)
    plt.close()

def get_distribution_strategy():
    # try:
    #     resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    #     print(f"Found TPU at: {resolver.master()}")
    #     tf.config.experimental_connect_to_cluster(resolver)
    #     tf.tpu.experimental.initialize_tpu_system(resolver)
    #     print("TPU system initialized successfully.")
    #     print("All TPU devices:", tf.config.list_logical_devices('TPU'))
    #     strategy = tf.distribute.TPUStrategy(resolver)
    #     print(f"Number of TPU cores: {strategy.num_replicas_in_sync}")
    #     return strategy
    # except ValueError:
    #     print("TPU not found. Checking for GPUs...")
    
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Running on {len(gpus)} GPUs")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        print("Running on single GPU")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        print("Running on CPU")
    
    return strategy

strategy = get_distribution_strategy()

data_dir = 'datasets/tcg_magic/data/train'
batch_size = 32 * strategy.num_replicas_in_sync

train_generator = CustomDataGenerator(data_dir, batch_size, augment=True)
validation_generator = CustomDataGenerator(data_dir, batch_size)

images, labels = train_generator.__getitem__(0)

augmented_visualization_path = 'datasets/tcg_magic/augmented_images.png'
save_image_visualization(images, labels, train_generator.classes, augmented_visualization_path)

num_samples = len(train_generator.image_files)
num_train = int(0.8 * num_samples)
train_generator.image_files = train_generator.image_files[:num_train]
validation_generator.image_files = validation_generator.image_files[num_train:]

def create_model(num_classes):
    with strategy.scope():
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model

with strategy.scope():
    model = create_model(len(train_generator.classes))

log_dir = "models/tcg_magic/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs("models/tcg_magic/logs", exist_ok=True)

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ModelCheckpoint('models/tcg_magic/best_model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

epochs = 100
print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=callbacks
)

model.save('models/tcg_magic/classifier.keras')

def unfreeze_model(model):
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

print("Preparing to fine-tune...")
with strategy.scope():
    model = unfreeze_model(model)

print("Starting fine-tuning...")
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=callbacks
)

print("Fine-tuning completed.")

model.save("models/tcg_magic/classifier_fine_tuned.keras")
print("Fine-tuned model saved.")

# Plotting training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('models/tcg_magic/training_history.png')
plt.close()

# Plotting fine-tuning history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_fine.history['accuracy'], label='Train Accuracy')
plt.plot(history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fine-tuned Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_fine.history['loss'], label='Train Loss')
plt.plot(history_fine.history['val_loss'], label='Validation Loss')
plt.title('Fine-tuned Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('models/tcg_magic/fine_tuning_history.png')
plt.close()