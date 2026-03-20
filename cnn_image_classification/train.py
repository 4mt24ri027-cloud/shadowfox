"""
ShadowFox AIML Internship - Task 1: Image Tagging
File: train.py

Description:
This script handles the actual learning process. It imports the data and
the model architecture, compiles the model with an optimizer and loss function,
trains the AI over multiple epochs, and saves the final "brain" as an .h5 file.
"""
import tensorflow as tf
from data_preparation import x_test,x_train,y_train,y_test,data_augmentation
from model import create_cnn_model

print("1. Assembling the complete AI...")
# Grab the blank brain from model.py
base_model = create_cnn_model()

# Attach the rotation/flipping pipeline to the very front of the brain
model = tf.keras.Sequential([
    data_augmentation,
    base_model
])

print("2. Compiling the multivariable calculus engine...")
# Set up the grading system and the optimizer
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("3. Starting the training process (This will take a few minutes!)...")
# Train the AI for 10 full passes (epochs) through the dataset
history = model.fit(x_train,y_train,epochs=30,validation_data=(x_test,y_test))

print("4. Saving the trained brain...")
# Save the physical weights to your hard drive
model.save('image_tagger_model.h5')
print("Success! The AI is fully trained and saved as 'image_tagger_model.h5'")


#Just for curious one
# The Grader's Rubric and Optimizer
# Note on Loss: Keras uses Cross-Entropy Math: Loss = -ln(p)
# To manually calculate the AI's confidence from the terminal output: p = e^(-Loss)
# (e.g., a printed Loss of 1.0 means the AI is 36.8% confident)