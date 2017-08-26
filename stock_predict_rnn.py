# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

sc = MinMaxScaler()

def preProcessStockData(in_set):
    # keep only stock open column (1) value
    tset = in_set.iloc[:,1:2].values
    tset_size = tset.size
    tset_size_min_1 = tset_size - 1
    # feature scaling
    # basic normalization between min/max.. resulting in value 0 to 1
    tset = sc.fit_transform(tset)

    # prepare x and y sets.. input set is daily open (hence we do not have last value)
    # output set is tomorrow's open (hence from 1 to last value)
    x_train = tset[0:tset_size_min_1]
    y_train = tset[1:tset_size]
    # Reshape the input - keras expects 3 dim of data
    # 1st dim - input data
    x_train = np.reshape(x_train, (tset_size_min_1, 1, 1))
    return (x_train, y_train)

# read train dataset
train_set = pd.read_csv('Google_Stock_Price_Train.csv')
x_train, y_train = preProcessStockData(train_set)
# initialize RNN
model = Sequential()

# Add input layer & LSTM layer
# use 4 memory units for best results (how far back to go for prev memory)
model.add(LSTM(units = 4, activation='sigmoid', input_shape=(None, 1)))

# Add the output layer with one output
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=32, epochs=200)

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))
predicted_stock_price = model.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color="red")
plt.plot(predicted_stock_price, color="blue")
plt.show()
