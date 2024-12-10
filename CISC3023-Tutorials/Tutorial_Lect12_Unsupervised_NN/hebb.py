# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:54:29 2023

@author: longchen
"""
import numpy as np

p = np.array([[0, 0, 0, 0, 0],
              [0, 1, 0, 0, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1]])

r = 5  # Five neurons in the input layer
s = 5  # Five neurons in the output layer
[r, Q] = p.shape

w = np.eye(s)   # 5x5 identity matrix. row = neuron.
b = np.random.rand(s, 1)    # Random vector with length of 5

max_epoch = 1000  # Maximum number of epochs
lr = 0.1         # Learning rate
dr = 0.02        # Decay rate or forgetting factor

for epoch in range(1, max_epoch+1):
    for q in range(Q):
        # Presentation phase: Neuron Activation, length-5 vector
        activation = np.heaviside(np.dot(w, p[:, q]).reshape(-1, 1) - b, 0)

        # Learning phase: Update weights with Hebb's Law
        dw = lr * np.outer(activation, p[:, q]) - dr * w
        w = w + dw

p = np.array([[1], [0], [0], [0], [1]])
activation = np.heaviside(np.dot(w, p).reshape(-1, 1) - b, 0)

for weights in w:
    print(weights.astype(np.float32))

print(f"Prediction res:\n {activation}")
