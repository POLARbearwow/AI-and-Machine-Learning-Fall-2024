import numpy as np

def generate_linear_data(a, b, size, noise=5):
    X = np.arange(size).reshape(size, 1)
    y = a * X + b + np.random.normal(0, noise, size=X.shape)
    y = y.reshape(-1)
    return X, y

def add_bias(X_train):
    return np.c_[np.ones((X_train.shape[0], 1)), X_train]
