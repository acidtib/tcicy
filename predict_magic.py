import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pathlib

# Load the trained model
model = tf.keras.models.load_model('models/tcg_magic/model.keras')

# Load class names from the text file
with open('models/tcg_magic/class_names.txt', 'r') as f:
    class_names = f.read().splitlines()

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    return img_array

# Function to predict the card name
def predict_card_name(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence

# Path to the image you want to scan
card_image_path = "datasets/tcg_magic/Forest/Forest_280_Bloomburrow.png"

# Perform prediction
predicted_class, confidence = predict_card_name(card_image_path)

print(f"This card most likely belongs to '{predicted_class}' with a {confidence:.2f}% confidence.")
