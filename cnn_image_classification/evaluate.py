"""
ShadowFox AIML Internship - Task 1: Image Tagging
File: evaluate.py

Description:
This script evaluates the trained model to calculate Accuracy,
Precision, Recall, and the F1-Score, fulfilling the Model Evaluation requirement.
"""
import tensorflow as tf
import numpy as np
from data_preparation import x_test,y_test
from sklearn.metrics import classification_report

print("1. Loading the trained AI brain...")
model = tf.keras.models.load_model('image_tagger_model.h5')

print("2. Giving the AI the 10,000 image final evaluation...")
# Get the raw mathematical scores from the model
raw_predictions = model.predict(x_test)

print("1. Loading the trained AI brain...")
model = tf.keras.models.load_model('image_tagger_model.h5')

print("2. Giving the AI the 10,000 image final exam...")
# Get the raw mathematical scores from the model
raw_predictions = model.predict(x_test)

# Convert raw scores to actual class guesses (0-9)
y_pred = np.argmax(tf.nn.softmax(raw_predictions), axis=1)

# The English labels for our report
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print("\n3. Generating the Official Performance Report...\n")
# Generate the classification report (Precision, Recall, F1)
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)