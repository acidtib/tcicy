import tensorflow as tf
import numpy as np
from PIL import Image
import os

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    with Image.open(image_path) as img:
        img = img.resize(target_size)
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_card(model_path, image_path, class_names):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    processed_image = load_and_preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image)
    
    # Get the predicted class index and confidence
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # Get the predicted class name (full card ID)
    predicted_card_id = class_names[predicted_class_index]
    
    return predicted_card_id, confidence

# Load class names (full card IDs)
def load_class_names(data_dir):
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    class_names = sorted([os.path.splitext(f)[0] for f in image_files])  # Use full filename without extension
    return class_names

# Example usage
if __name__ == "__main__":
    model_path = 'models/tcg_magic/classifier_fine_tuned.keras'  # Path to your trained model
    data_dir = 'datasets/tcg_magic/training'  # Path to your training data directory
    test_image_path = "datasets/tcg_magic/training/00adcfec_9893_48e7_b905_158eac5497f2.png"
    # test_image_path = "/home/acid/Downloads/IMG_1460.jpg"
    
    class_names = load_class_names(data_dir)
    
    predicted_card_id, confidence = predict_card(model_path, test_image_path, class_names)
    
    print(f"Predicted Card ID: {predicted_card_id}")
    print(f"Confidence: {confidence:.2f}")
  