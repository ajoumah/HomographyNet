# 🖼️ HomographyNet

## Deep Learning for Planar Homography Estimation 🎯

---

### 🚀 Overview

**HomographyNet** is a deep convolutional neural network that estimates planar homographies directly from pairs of grayscale image patches. It predicts the 8 parameters representing the corner shifts of a warped image patch, enabling applications like image alignment, perspective correction, and geometric transformations — essential tools in computer vision and pattern recognition.

This model learns end-to-end from data, making it robust to noise and variations in images.

---

### 👤 Author

**Ahmad El Jouma**  
📅 January 2021

---

### ⚙️ Features

- 🎛️ **Convolutional Neural Network** tailored for homography regression
- 📊 **Predicts 4-point corner displacements** (8 parameters total)
- 🔄 **Data augmentation** with random perturbations for robust training
- 📚 **Training data generation** from standard image datasets (e.g., COCO)
- 🖥️ **Visualization tools** for evaluating predictions and transformations
- 💾 Save and load model weights for flexible training and inference
- 🔧 Custom **Euclidean distance loss** for accurate parameter regression

---

### 📦 Requirements

- Python 3.6+
- [Keras](https://keras.io) (TensorFlow backend)
- [TensorFlow](https://www.tensorflow.org)
- [OpenCV](https://opencv.org) (`opencv-python`)
- NumPy
- Matplotlib

Install dependencies via:

```bash
pip install keras tensorflow opencv-python numpy matplotlib
