"""
ShadowFox AIML Internship - Task 1: Image Tagging (Deployment)
File: app.py

Description:
This script serves as the production-grade deployment for our trained CNN model.
It uses FastAPI to create a high-speed, asynchronous backend API. The server
receives images via HTTP POST requests, forcefully reshapes them to 32x32 pixels,
runs the multivariable calculus prediction, and returns the results as JSON data.

Author: Prajnesh
"""
from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import csv
from datetime import datetime

# ==========================================
# 1. INITIALIZE API & SECURITY
# ==========================================
app = FastAPI(title="Vision AI API", description="Production API for CNN Image Tagging")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. TELEMETRY SETUP (Logging System)
# ==========================================
LOG_DIR = "telemetry_logs"
IMG_DIR = os.path.join(LOG_DIR, "saved_images")
CSV_FILE = os.path.join(LOG_DIR, "prediction_logs.csv")

# Create folders and CSV headers if they don't exist yet
os.makedirs(IMG_DIR, exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Filename", "Prediction", "Confidence"])


# ==========================================
# 3. LOAD THE AI BRAIN
# ==========================================
# This sits outside the route so the heavy .h5 file only loads ONCE at startup
model = tf.keras.models.load_model('image_tagger_model.h5')
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# ==========================================
# 4. THE API ENDPOINT (The Data Bridge)
# ==========================================
@app.post("/predict/")
async def predict_image(file: UploadFile=File(...)):
    # --- A. Read and Preprocess the Image ---
    # Extract the digital bytes and convert them to an RGB image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # The 32x32 Squish and normalization for the CNN
    img_resized = image.resize((32, 32))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- B. Run the Prediction Math ---
    raw_logits = model.predict(img_array, verbose=0)
    probabilities = tf.nn.softmax(raw_logits).numpy()

    winning_number = np.argmax(probabilities, axis=1)[0]
    winning_word = class_names[winning_number]
    confidence = float(np.max(probabilities) * 100)

    # --- C. Telemetry: Save Data & Image ---
    # Create a unique timestamp so files don't overwrite each other
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"{current_time}_{file.filename}"

    # Save the physical picture to the hard drive
    img_save_path = os.path.join(IMG_DIR, saved_filename)
    image.save(img_save_path)

    # Append the prediction math to the bottom of the CSV spreadsheet
    with open(CSV_FILE, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([current_time, saved_filename, winning_word.upper(), f"{confidence:.2f}"])

    # --- D. Return Response to User ---
    # Send the final results back to the website as a lightweight JSON dictionary
    return {
        "filename": file.filename,
        "prediction": winning_word.upper(),
        "confidence_score": f"{confidence:.2f}%"
    }