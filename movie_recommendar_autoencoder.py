# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Read all the data and pre-process data
# read movie, user, and rating info
movies = pd.read_csv('/home/osboxes/rbm/ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('/home/osboxes/rbm/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('/home/osboxes/rbm/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# read train and test set and convert them to array from panda DataSet
train_set = pd.read_csv('/home/osboxes/rbm/ml-100k/u1.base', delimiter='\t')
train_set = np.array(train_set, dtype='int')
test_set = pd.read_csv('/home/osboxes/rbm/ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Get max id of users and movies across train/test datasets
nb_users = int(max(max(train_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(train_set[:,1]), max(test_set[:,1])))

# convert incoming data to 2D - row = user and column = movie and each cell as rating
# it's max dimension would be nb_users x nb_movies
def convert(data):
    new_data = []
    for id_user in range(1, nb_users+1):
        # pick user movies and ratings
        id_movies = data[:,1][data[:,0] == id_user]
        id_ratings = data[:,2][data[:,0] == id_user]
        # create ratings for all movies (not just user rated movies)
        ratings = np.zeros(nb_movies)
        # now populate user movie ratings
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

train_set = convert(train_set)
test_set = convert(test_set)

# use list of list train and test sets to torch Tensors
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

# create the architecture of autoencoder neural network
# Stacked AutoEncoder class inheriting from nn.Module
class SAE(nn.Module):
    
    def __init__(self, ):
        # initialize properties inherited from parent by callling parent init
        super(SAE, self).__init__()
        # create first hidden layer with nb_movies input and 20 output
        # 20 outputs could represent any extracted features (genre, director,...)
        self.fc1 = nn.Linear(nb_movies, 20)
        # second layer with 20 input and 10 output
        self.fc2 = nn.Linear(20, 10)
        # since it's symmetric create 3rd and 4th layer mirror of 1st and 2nd
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        # activation function is going to be sigmoid
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        # pass input through each layer and use activation function in between
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        # since it's last layer, we do not need to apply activation func
        # as there is no other subsequent layers
        x = self.fc4(x)
        return x
    
sae = SAE()
# we will use mean squared error func to evaluate predicted value with actual value
criterion = nn.MSELoss()
# use learning rate of 0.1
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Training the SAE
nb_epoch = 50
for epoch in range(1, nb_epoch+1):
    train_loss_sum = 0
    train_count = 0.
    for id_user in range(nb_users):
        # pick current user rating and remove rating=0 (non rated entries)
        input = Variable(train_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss_sum += np.sqrt(loss.data[0] * mean_corrector)
            train_count += 1.
            optimizer.step()
    print("Finished epoch {} with avg loss {}".format(epoch, train_loss_sum/train_count))
    
# now test the model
test_loss_sum = 0
test_count = 0.
for id_user in range(nb_users):
    # pick current user rating and remove rating=0 (non rated entries)
    input = Variable(train_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss_sum += np.sqrt(loss.data[0] * mean_corrector)
        test_count += 1.
print("Finished testing {} with avg loss {}".format(epoch, test_loss_sum/test_count))
