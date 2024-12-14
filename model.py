import os
import tensorflow as tf
import numpy as np

# Define constants
# MODEL_PATH = "model.keras"  # Path to your Keras model4
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.keras')

IMG_SIZE = 28              # Replace with the size used during training
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'sub', 'mul', 'div']  # Modify based on your classes

# Load the model
def load_trained_model():
    """
    Load the pre-trained Keras model from the file system.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model


# Map class index to label
def get_label_mapping():
    """
    Returns a dictionary mapping indices to class labels.
    """
    return {i: label for i, label in enumerate(LABELS)}

# Prediction utility
def predict_symbol(model, symbol_image):
    """
    Predict the class of a single preprocessed symbol image.

    Args:
    - model: Loaded Keras model.
    - symbol_image: Preprocessed symbol image (normalized, resized).

    Returns:
    - Tuple of (predicted_label, probability_distribution)
    """
    symbol_image = np.expand_dims(symbol_image, axis=0)  # Add batch dimension
    predictions = model.predict(symbol_image, verbose=0)  # Predict
    predicted_index = np.argmax(predictions)
    label_mapping = get_label_mapping()
    predicted_label = label_mapping[predicted_index]
    return predicted_label, predictions[0]