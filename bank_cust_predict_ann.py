# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

# Part 1 - Reading and pre-processing data
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

le1 = LabelEncoder()
x[:, 1] = le1.fit_transform(x[:, 1])
le2 = LabelEncoder()
x[:, 2] = le2.fit_transform(x[:, 2])

enc = OneHotEncoder(categorical_features=[1])
x = enc.fit_transform(x).toarray()

x = x[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Part 2 = Build ANN

# Initialize ANN
classifier = Sequential()

# Add the input layer and first hidden layer
# input dim = 11 since there are 10 features + 1 theta(0)
# hidden layer size = 6
# activation function is - relu (rectifier func)
# kernel init - initialize weights with random uniform (between 0, 1)
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Add the second layer
# input dim is implied as the output of first hidden layer
# output dim for this layer is 6
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

# Add final output layer
# Ouput dim is 1.. as there is only one output
# activation function is sigmoid (as we're looking for probability as one output)
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

#now that ANN layers are added, compile the net
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# now train the ANN with train dataset
classifier.fit(x_train, y_train, batch_size=10, epochs=15)

# Part 3 - Making the predictions and evaluating the model
y_pred = classifier.predict(x_test)
# convert probability to boolean outcome
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
