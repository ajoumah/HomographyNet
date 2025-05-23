# HomographyNet

Deep Homography Estimation for Pattern Recognition

---

## Overview

HomographyNet is a deep learning model designed to estimate planar homographies between image patches. It predicts the 4-point homography parameters directly from pairs of grayscale image patches, enabling tasks such as perspective correction, image alignment, and geometric transformations useful in computer vision and pattern recognition applications.

This implementation is based on a convolutional neural network architecture that regresses the eight parameters representing corner displacements of a warped patch.

---

## Author

**Ahmad El Jouma**  
January 2021

---

## Features

- Deep homography regression network using Keras and TensorFlow backend.
- End-to-end learning of 4-point homography parameters.
- Custom Euclidean distance loss for precise homography estimation.
- Data loader and generator for training on COCO dataset image patches.
- Visualization of original and warped images for evaluation.
- Save and load model weights for training continuation or inference.
- Supports batch training with data augmentation via random perturbations.

---

## Requirements

- Python 3.6+
- Keras
- TensorFlow (backend for Keras)
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

Install required packages using:

```bash
pip install keras tensorflow opencv-python numpy matplotlib
