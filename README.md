# Traffic Sign AI Trainer (TensorFlow CNN)

## Overview

This project is a deep learning-based image classification system built using TensorFlow and Keras. It is designed to recognize and classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

The system uses a Convolutional Neural Network (CNN) to learn visual patterns from images and classify them into 43 distinct traffic sign categories.

---

## How It Works

### 1. Data Loading & Preprocessing
- Images are loaded from a structured directory where each folder represents a class label.
- Each image is resized to **32x32 pixels**.
- Images are converted into numerical arrays and normalized to a range of **0 to 1**.
- The dataset is shuffled to ensure randomness.
- Data is split into:
  - 80% training set
  - 20% testing set

---

### 2. Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of:

- Multiple **Convolutional Layers** for feature extraction
- **MaxPooling Layers** for dimensionality reduction
- **Dropout Layers** to reduce overfitting
- A **Flatten Layer** to convert feature maps into a vector
- A **Dense Fully Connected Layer** for learning complex patterns
- A **Softmax Output Layer** for multi-class classification (43 classes)

---

### 3. Training Process

- The model is compiled using:
  - Optimizer: Adam
  - Loss Function: Sparse Categorical Crossentropy
  - Metric: Accuracy

- The training process:
  - Feeds training images into the model
  - Validates performance using test data
  - Uses TensorBoard for monitoring metrics

---

### 4. Model Output

The trained model predicts the probability of an input image belonging to one of **43 traffic sign classes**, such as:

- Stop signs
- Speed limits
- Warning signs
- Direction signs

The highest probability class is selected as the final prediction.

---

### 5. Model Saving

After training, the model is saved in HDF5 format for later use:


model/Traffic_detection.h5


---

## Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Scikit-learn

---

## How to Run

```bash
python train.py
Project Purpose

The main goal of this project is to demonstrate how deep learning can be used for real-world image classification tasks such as traffic sign recognition, which is an important component in autonomous driving systems.

Future Improvements
Add data augmentation for better generalization
Use advanced architectures (e.g., ResNet, MobileNet)
Improve dataset balancing
Deploy as a Flask or FastAPI web service
