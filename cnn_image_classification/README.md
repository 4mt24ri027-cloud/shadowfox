
# 👁️ Vision AI Dashboard

### End-to-End CNN Image Classification & Real-Time Deployment

---

## 🚀 Overview

**Vision AI Dashboard** is a full-stack machine learning application that performs real-time image classification using a custom Convolutional Neural Network (CNN). The system is trained on the CIFAR-10 dataset and deployed using a high-performance API, enabling seamless interaction between deep learning models and a modern web interface.

This project demonstrates the complete ML lifecycle—from data preprocessing and model training to production deployment and live monitoring.

---

## ✨ Features

* 🧠 **Custom CNN Model**
  Built and trained from scratch for multi-class image classification.

* ⚡ **FastAPI Backend**
  Asynchronous API for fast and scalable inference.

* 📊 **Telemetry Logging**
  Logs predictions, confidence scores, and timestamps into CSV files for monitoring.

* 🖼️ **Image Archiving**
  Automatically stores uploaded images for traceability and debugging.

* 🌓 **Modern UI Dashboard**
Glassmorphism-inspired interface (UI 
boilerplate generated via AI assistance).
---

## 🛠️ Tech Stack

| Category       | Tools Used                        |
| -------------- | --------------------------------- |
| Language       | Python                            |
| ML Framework   | TensorFlow, Keras                 |
| Data Handling  | NumPy, Scikit-learn               |
| Backend        | FastAPI, Uvicorn                  |
| Image Handling | Pillow                            |
| Frontend       | HTML, CSS, JavaScript (Fetch API) |

---

## 📂 Project Structure

```
├── data_preparation.py    # Load and preprocess dataset
├── model.py               # CNN architecture
├── train.py               # Model training script
├── evaluate.py            # Performance evaluation
├── prediction.py          # Local testing
├── deployment.py          # FastAPI server
├── image_tagger_model.h5  # Trained model
├── index.html             # Frontend dashboard
├── requirements.txt       # Dependencies
└── telemetry_logs/        # Prediction logs (auto-generated)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/vision-ai-dashboard.git
cd vision-ai-dashboard
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

To train the CNN model:

```bash
python train.py
```

This will:

* Load CIFAR-10 dataset
* Train the CNN
* Save model as `image_tagger_model.h5`

---

## 📊 Model Evaluation

```bash
python evaluate.py
```

Outputs:

* Accuracy & Loss
* Classification Report

---

## 🔮 Run Predictions Locally

```bash
python prediction.py
```

---

## 🚀 Run the API Server

```bash
uvicorn deployment:app --reload
```

API will be available at:
👉 http://127.0.0.1:8000

Interactive Docs (Swagger UI):
👉 http://127.0.0.1:8000/docs

---

## 🌐 Frontend Usage

1. Open `index.html` in your browser
2. Upload an image
3. View:

   * Predicted class
   * Confidence score
   * Live logs

---

## 📈 Telemetry Logging

Each prediction is logged with:

* Filename
* Predicted label
* Confidence score
* Timestamp

Logs are stored in:

```
telemetry_logs/logs.csv
```

---

## 🧩 Future Improvements

* 🔥 Transfer Learning (ResNet, MobileNet)
* 🐳 Docker Deployment
* ☁️ Cloud Hosting (AWS/GCP)
* 📊 Advanced Monitoring Dashboard
* 🔐 Input Validation & Security Enhancements

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* CIFAR-10 Dataset
* TensorFlow & Open Source Community

---

## 👨‍💻 Author

**Prajnesh **

* GitHub: https://github.com/4mt24ri027-cloud
* LinkedIn: www.linkedin.com/in/prajnesh-shetty-208515385



---

⭐ If you found this project useful, consider giving it a star!
