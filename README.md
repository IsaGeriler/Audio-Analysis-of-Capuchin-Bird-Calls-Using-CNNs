# 🦜 Audio Analysis of Capuchin Bird Calls Using CNNs

## 📌 Project Overview
This project applies **Convolutional Neural Networks (CNNs)** to classify and analyze **Capuchin bird calls** from audio recordings. By leveraging deep learning techniques, the model enables automated **bioacoustic analysis** to aid conservation efforts.

## 🚀 Features
- **Audio Preprocessing:** Converts raw recordings to spectrograms using **Short-Time Fourier Transform (STFT)**.
- **Deep Learning Model:** Implements a **CNN architecture** for pattern recognition.
- **Data Augmentation:** Enhances robustness with **white noise, filtering, and normalization**.
- **High Accuracy:** Achieves **98% classification accuracy** on test data.
- **Evaluation Metrics:** Uses **precision, recall, confusion matrix, and ROC curves**.

## 🔧 Technologies Used
- **Python**
- **TensorFlow/Keras**
- **Scikit-learn**
- **Google Colab**
- **Pydub**
- **NumPy & Matplotlib**

## 📂 Dataset
- Pre-existing **Capuchin bird audio recordings**.
- Data split: **70% training, 15% validation, 15% test**.
- **Augmentation techniques**: White noise, high-pass & low-pass filtering.

## ⚙️ Model Architecture
- **Convolution Layers:** Extract spectral features from spectrograms.
- **Max Pooling:** Reduces dimensionality for efficient learning.
- **Dropout Layers:** Prevents overfitting.
- **Dense Layer with Sigmoid Activation:** Outputs classification probability.
- **Adam Optimizer & Binary Crossentropy Loss:** Ensures optimal training.

## 📊 Results
- **Accuracy:** 98%
- **Precision:** 98%
- **Recall:** 97%
- **Confusion Matrix:** Shows effective classification of bird calls.
- **Comparison with Other Models:** Outperforms existing methods in noise resistance.

## 📖 Future Improvements
- Enhancing noise filtering for overlapping signals.
- Training on larger, more diverse datasets.
- Optimizing for real-time bird call classification.

## 📜 Citation
If you use this project, please cite:
> Audio Analysis of Capuchin Bird Calls Using CNNs by **Isa Berk Geriler & Duygu Önder**.

---
🔗 **GitHub Repository:** https://github.com/IsaGeriler/Audio-Analysis-of-Capuchin-Bird-Calls-Using-CNNs
