import numpy as np
import random

# Seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

# Generate dataset
X = np.array([2 * np.random.random() for _ in range(100)])
y = 4 + 3 * X + np.random.normal(0, 1, size=X.shape)

# Split into training and test sets
train_size = int(X.size * 0.8)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

print(f"trainSize: {train_size}")
print(f"X_train.size: {X_train.size}; X_train: {X_train}")
print(f"X_test.size: {X_test.size}; X_test: {X_test}")
print(f"y_train.size: {y_train.size}; y_train: {y_train}")
print(f"y_test.size: {y_test.size}; y_test: {y_test}")

# Add bias term to X_train and X_test
X_train_b = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_b = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialize weights
theta = np.random.randn(2)

# Learning rate
learning_rate = 0.01
n_iterations = 1000
m = X_train.size

# Gradient Descent
for iteration in range(n_iterations):
    y_pred = X_train_b.dot(theta)
    errors = y_train - y_pred

    gradients = -2.0 / m * X_train_b.T.dot(errors)
    theta -= learning_rate * gradients

# Make predictions on the test set
y_test_pred = X_test_b.dot(theta)

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_test - y_test_pred) ** 2)

print(f"Theta: {theta}")
print(f"Mean Squared Error (MSE): {mse}")
