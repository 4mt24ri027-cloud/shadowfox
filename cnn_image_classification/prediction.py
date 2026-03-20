"""
ShadowFox AIML Internship - Task 1: Image Tagging
File: predict.py

Description:
This script wakes up our trained AI, downloads a random picture from the internet,
shrinks it to fit our custom "Eyes" (32x32), and asks the AI to classify it.
"""

"""
ShadowFox AIML Internship - Task 1
File: build_test_folder.py

Description: This script creates a folder called 'test_images', disguises itself 
as a web browser to bypass firewalls, and automatically downloads 10 perfect test images.
"""

import os
import tensorflow as tf
import numpy as np

#Translation dictionary
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print("Waking up the AI.....")
model = tf.keras.models.load_model('image_tagger_model.h5')

folder_name = 'test_images'
print("------prediction start------")
#loop through every file name in folder
for filename in sorted(os.listdir(folder_name)):

    # We only want to process image files, not random hidden system files
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):

        # Connect the folder name and file name together to get the exact path
        image_path = os.path.join(folder_name,filename)

        # 1. Load the image and shrink it to fit the AI's 32x32 "eyes"
        img = tf.keras.utils.load_img(image_path,target_size=(32,32))

        # 2. Convert the picture into a 3D grid of raw numbers
        img_array = tf.keras.utils.img_to_array(img)

        # 3. Normalize the numbers to be between 0.0 and 1.0
        img_array = img_array / 255.0

        # 4. Put the image into a "batch" of 1 so Keras doesn't crash
        img_array = np.expand_dims(img_array, axis=0)

        # 5. The Prediction (Get raw Logits)
        raw_logits = model.predict(img_array, verbose=0)

        # 6. Apply Softmax to turn raw numbers into 0.0 - 1.0 probabilities
        predictions = tf.nn.softmax(raw_logits).numpy()

        # 7. Reading the Results (Now the math is correct!)
        winning_number = np.argmax(predictions)
        winning_word = class_names[winning_number]
        confidence = np.max(predictions) * 100

        # 8. Printing the Final Grade
        print(f"File: {filename.ljust(15)} | AI Guesses: {winning_word.upper()} | Confidence: {confidence:.1f}%")

