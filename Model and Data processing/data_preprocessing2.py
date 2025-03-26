import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import json

# Class decoder map (based on your emotion_map)
emotion_decoder = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fear', 6: 'disgust', 7: 'surprise'
}

# Decode function
def decode_emotion(one_hot_vector):
    index = np.argmax(one_hot_vector)
    return emotion_decoder.get(index, "unknown")

# Load preprocessed dataset
def load_preprocessed_data(path="merged_X.npy", label_path="merged_y.npy"):
    X = np.load(path, allow_pickle=True)
    y = np.load(label_path, allow_pickle=True)

    # Ensure the labels are integers (if not already)
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"Labels should be integers, but got {y.dtype}")
    
    return X, y

# Prepare data: one-hot encode labels and split
def prepare_data(X, y, test_size=0.2, random_state=42):
    # Ensure labels are integers before encoding
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"Labels should be integers, but got {y.dtype}")
    
    # One-hot encode labels
    y_encoded = to_categorical(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train X: {X_train.shape}, Train y: {y_train.shape}")
    print(f"Test X: {X_test.shape}, Test y: {y_test.shape}")
    
    # Number of classes based on one-hot encoded labels
    num_classes = y_encoded.shape[1]

    return X_train, X_test, y_train, y_test, num_classes

# Example usage
if __name__ == "__main__":
    X, y = load_preprocessed_data()

    print("Loaded merged dataset:")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test, num_classes = prepare_data(X, y)

    print(f"\nNumber of classes: {num_classes}")

    # Test decoding
    print("\nExample decoded test labels:")
    for i in range(3):
        # Pass the one-hot encoded vector to decode_emotion
        print(f"Encoded: {y_test[i]} ➤ Decoded: {decode_emotion(y_test[i])}")