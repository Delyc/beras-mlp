#  Beras – From-Scratch Deep Learning Framework

Implementation of a **Keras-like deep learning framework** called **Beras**, built entirely from scratch using **NumPy**.  
It supports automatic differentiation, dense layers, activation functions, loss functions, and optimizers — and is used to train a **Multi-Layer Perceptron (MLP)** on the **MNIST** handwritten digits dataset.

---

## Overview
This project was built as part of **Brown University’s CSCI 2470: Deep Learning** course.  
It reconstructs the core ideas behind TensorFlow and Keras by implementing each component manually — from tensors and gradients to optimizers and model training.

---

### Model Overview
This project re-implements the foundation of a deep learning library through a collection of modular files inside the `beras/` directory.

The **Beras framework** includes:
- A custom `Tensor` class with gradient tracking  
- A `GradientTape` system for auto-differentiation  
- `Dense` layers with activations (ReLU, Sigmoid, Softmax)  
- Loss functions (MSE, CategoricalCrossEntropy)  
- Optimizers (SGD, Adam)  
- Accuracy metrics and one-hot encoding utilities  

Once the framework is complete, it is used in `train.py` to train an **MLP** for **digit classification on MNIST**.

---

## Features
- Custom **Tensor** object supporting backpropagation  
- Fully manual **forward** and **backward** passes  
- Implementation of **Dense**, **Activation**, and **Loss** layers  
- Custom **optimizers** with weight updates (SGD, Adam)  
- **GradientTape** for auto-differentiation  
- **Preprocessing** utilities for MNIST (normalization + one-hot encoding)  
- End-to-end training pipeline for handwritten digit recognition  

---

## Dataset
This project uses the public **MNIST** dataset — 60,000 training and 10,000 test images of handwritten digits (0–9).  
Images are automatically downloaded via TensorFlow utilities in the preprocessing step and normalized to `[0, 1]`.

No manual download required.

---

# How to run the project
### Install dependencies
```
pip install -r requirements.txt
```

### Run training
```
python train.py
```
