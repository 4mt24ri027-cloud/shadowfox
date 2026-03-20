"""
ShadowFox AIML Internship - Task 1: Image Tagging
File: model.py

Description:
This script contains the blueprint for our Convolutional Neural Network (CNN).
It uses the TensorFlow Keras API to build a Sequential model architecture
designed specifically to extract features and classify images.
"""

from tensorflow.keras import models ,layers

def create_cnn_model():
    # 1. Initialize the Keras Sequential "Assembly Line"
    model = models.Sequential()

    # --- THE EYES (Feature Extraction) ---
    # First Block: Look at the raw image
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())  ##Hypertuning after the testing
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.2))

    # Second Block: Look for more complex patterns
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())  #hypertuning
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(.3))

    # Third Block: Final high-level feature extraction
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # --- THE BRAIN (Classification) ---
    # Convert 2D image data into a 1D list of numbers
    model.add(layers.Flatten())

    # The hidden logic layer
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    # The final output layer (10 categories)
    model.add(layers.Dense(10))

    return model

if __name__ == "__main__":
    test_model = create_cnn_model()
    test_model.summary()