"""
ShadowFox AIML Internship - Task 1: Image Tagging
File: data_preparation.py

Description:
This script downloads the CIFAR-10 image dataset, normalizes the pixel
values, and builds a data augmentation pipeline (rotation and flipping)
using TensorFlow and Keras to prevent model overfitting.
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# 1. Load the dataset directly from Keras
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# 2. Normalize the pixels (Convert from 0-255 to 0.0-1.0)
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

# 3. Build the Data Augmentation Pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1)
])
print("Data successfully loaded, normalized, and augmentation pipeline created!")