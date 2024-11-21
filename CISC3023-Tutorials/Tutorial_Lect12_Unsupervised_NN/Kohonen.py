# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:37:19 2023

@author: longchen
"""
import math


class SOM:
	def winner(self, weights, sample):
		"""
		Computes winning vector by Euclidean distance.
		Arguments:
			weights: All the weights.
			sample: Incoming data sample.
		Returns:
			The winner neuron.
		"""

		# Two distances
		d0, d1 = 0, 0

		for i in range(len(sample)):

			d0 += math.pow((sample[i] - weights[0][i]), 2)
			d1 += math.pow((sample[i] - weights[1][i]), 2)

		# Selecting the cluster with the smallest distance as winning cluster
		if d0 < d1:
			return 0
		else:
			return 1

	def update(self, weights, sample, J, alpha):
		"""
		Updates the winning vector.

		Arguments:
			weights: All weights
			sample: Data sample that causes the update.
			J: Which is the winner?
			alpha: Learning rate

		Returns:
			The upated weights.
		"""
		# Here iterating over the weights of winning cluster and modifying them
		for i in range(len(weights[0])):
			weights[J][i] = weights[J][i] + alpha * (sample[i] - weights[J][i])

		return weights


# Training Examples ( m, n ): 4 neurons in the input layer.
T = [
	[1, 1, 0, 0],
	[0, 0, 0, 1],
	[1, 0, 0, 0],
	[0, 0, 1, 1]
]

m, n = len(T), len(T[0])

# weight initialization ( n, C ): 2 neurons in the Kohonen layer.
weights = [
	[0.2, 0.6, 0.5, 0.9],		# Neuron 1
	[0.8, 0.4, 0.7, 0.3]		# Neuron 2
]

# training
ob = SOM()

epochs = 3
alpha = 0.5		# Learning Rate

for i in range(epochs):
	for j in range(m):
		# training sample
		cur_sample = T[j]

		# Compute winner vector
		J = ob.winner(weights, cur_sample)

		# Update winning vector
		weights = ob.update(weights, cur_sample, J, alpha)

# classify test sample
s = [0, 0, 0, 1]
J = ob.winner(weights, s)

print("Test Sample s belongs to Cluster : ", J)
print("Trained weights : ", weights)



