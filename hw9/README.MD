# Homework Assignment 09

## Assignment: Dimensionality Reduction and Autoencoders

### Objective:
In this assignment, you will explore dimensionality reduction using Principal Component Analysis (PCA) and simple Autoencoders (both linear and non-linear) on the Wine dataset. You will implement these algorithms in Python using NumPy and compare their reconstruction errors.

### Dataset:
The Wine dataset can be found at the UCI Machine Learning Repository: [Wine Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data). Please download the dataset and load it into your Python environment.

---

### Tasks:

#### 1. Implement PCA:
- Write a Python function using NumPy to perform PCA on the Wine dataset.
- Consider only the first two principal components for dimensionality reduction.
- Reconstruct the data using these two principal components.
- Visualize the principal components in the two-dimensional space.

#### 2. Train a Linear Autoencoder:
- Implement a linear autoencoder using a neural network design.
- Train the autoencoder on the Wine dataset.
- Reconstruct the data using the trained autoencoder.
- Visualize the output of the encoder; calculate and report the reconstruction error.

#### 3. Train a Non-linear Autoencoder:
- Implement a non-linear autoencoder with at least one hidden layer that uses an activation function (e.g., ReLU).
- Train the autoencoder on the Wine dataset.
- Reconstruct the data using the trained non-linear autoencoder.
- Visualize the output of the encoder; calculate and report the reconstruction error.

### 4. Comparison and Analysis:
   - Compare the reconstruction errors of the PCA, linear autoencoder, and non-linear autoencoder.
   - Analyze the results and discuss which method provides the best reconstruction accuracy and why.