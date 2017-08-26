# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ccdata = pd.read_csv('Credit_Card_Applications.csv')
# separate out the last column.. it is to indicate if the CC application was approved or not
x = ccdata.iloc[:, :-1].values
y = ccdata.iloc[:, -1].values

# scale all the data between 0 & 1
# no need to transform y.. as it is already 0 & 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x)

# training the SOM
# we'll use the minosom library
from minisom import MiniSom

# output grid = 10x10
# feature (columns in the CC data) = 15
som = MiniSom(x = 10, y = 10, input_len=15, random_seed=1)
# randomly initilize the weights..
som.random_weights_init(x)
# begin the training on input data x
som.train_random(x, num_iteration=100)

# now that the SOM has been trained, plot the 2D map
from pylab import bone, pcolor, colorbar, plot, show
# empty graph panel
bone()
# plot color for the 2D map
# pick up 2D map from SOM and transpose the matrix
distance_map = som.distance_map()
pcolor(distance_map.T)
# add the color bar on the side to indicate color mapping to the value
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
# iterate through each customer (row) and it's features (columns)
for f, cust in enumerate(x):
    # find the winner node in 10x10 grid for the customer
    w = som.winner(cust)
    # plot green/red marker at the winner grid location
    # we'll use color/marker based on y (weather customer got approval or not)
    plot(w[0] + 0.5, w[1] + 0.5,
         markers[y[f]],
         markeredgecolor=colors[y[f]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# now we extract fraud customers from the SOM (focus on the ones that got approved)
mappings = som.win_map(x)
# looking into the plot, we know that grid location 7,1 & 8,4 are white (value 1) and 
# are outliers and hence should be examined for fraud.. we could potentially include grid
# location with values above certain threshold (say 0.98).. but for now only 1s..
fraud_loc1 = mappings[7, 1]
fraud_loc2 = mappings[8, 4]
frauds = np.concatenate((fraud_loc1, fraud_loc2), axis=0)
# list of customers who could potentially be frauds.. examines the ones who got their 
# credit card approved.. correlate fraud customer id to y list above and narrow down
# the list to only approved fraud suspects
frauds = sc.inverse_transform(frauds)
