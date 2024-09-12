# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:42:15 2024

@author: longchen
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
data = pd.read_csv(url, header=None, delimiter=',', na_values='?')
data.columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

# dropping missing values column
data.drop(['stalk-root'], axis=1, inplace=True)


# Convert categorical variables using OrdinalEncoding
oe = OrdinalEncoder()       # Convert into numerical label.
X = data.drop('class', axis=1).to_numpy()
y = data['class'].to_numpy()
X = oe.fit_transform(X)
y = oe.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical NB model
clf = CategoricalNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Categorical Naive Bayes Accuracy:", accuracy)
