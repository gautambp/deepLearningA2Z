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

# convert incoming data to 2D - row = user and column = movie
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

# since we're expecting binary response (1 = liked movie & 0 = did not like movie)
# convert all ratings (1, 2 3, 4, 5) to binary response
# since 0 indicates not rated.. conver it to -1
def convertRating(data):    
    data[data == 0] = -1
    data[data == 1] = 0
    data[data == 2] = 0
    data[data >= 3] = 1
    return data

train_set = convertRating(train_set)
test_set = convertRating(test_set)

# create class to build RBM architecture
class RBM():
    # constructor takes in nv - # of visible nodes and nh - # of hidden nodes
    def __init__(self, nv, nh):
        # initialize weights/bias to random vales with normal distribution
        # we need bias for hidden and visible nodes..
        self.w = torch.randn(nh, nv)
        self.bh = torch.randn(1, nh)
        self.bv = torch.randn(1, nv)
        
    def sample_h(self, x):
        # basic computation - sigmoid(x * w + bh)
        wx = torch.mm(x, self.w.t())
        activation = wx + self.bh.expand_as(wx)
        # probability of hidden node given visible node
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        # basic computation - sigmoid(x * w + bv).. but in reverse
        # going from visible node to hidden node
        wy = torch.mm(y, self.w)
        activation = wy + self.bv.expand_as(wy)
        # probability of hidden node given visible node
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    # train function - based on contrastive divergence algo for RBM
    # inputs - v0 - current row (current user rating) that you're working on
    # vk - value of visible node after k iteration
    # ph0 - current row probability of hidden node to be 1 given v0
    # phk - probability after k iteration
    def train(self, v0, vk, ph0, phk):
        self.w += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        # adding with 0 to maintain 2D structure
        self.bv = torch.sum((v0 - vk), 0)
        self.bh = torch.sum((ph0 - phk), 0)

# input (visible) nodes are # of movies        
nv = len(train_set[0])
# hidden layer is about how many features we want to extract
# it could be genre, director, actor, sub-genre,...
nh = 100
# we do not update weights for each row.. rather we do it after the batch inputs
batch_size = 100
# no of epochs
nb_epoch = 10

rbm = RBM(nv, nh)

for epoch in range(1, nb_epoch+1):
    train_loss_sum = 0
    train_count = 0
    for id_user in range(0, nb_users-batch_size, 100):
        # pick first batch as initial values for the visible node
        v0 = train_set[id_user:id_user+batch_size]
        # since k=0 (first iteration) - vk = v0
        vk = train_set[id_user:id_user+batch_size]
        # derive initial probability for hidden node h given v by sampling at k=0
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            # take prob for hidden node for k based on kth visible nodes
            _, hk = rbm.sample_h(vk)
            # update kth visible nodes based on kth computed hidden node values for prev k
            _, vk = rbm.sample_v(hk)
            # maintain the -1 ratings in vk.. since v0 is not modified, we can pick it up from there
            vk[v0<1] = v0[v0<1]
        # derive probability for hidden node h given v by sampling at k=last iteration
        phk, _ = rbm.sample_h(vk)
        # now train the model.. update weights and biases
        rbm.train(v0, vk, ph0, phk)
        # compute the error
        train_loss_sum += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        train_count += 1
    print("Finished epoch {} with avg loss {}".format(epoch, train_loss_sum/train_count))

# Testing RBM with test data
test_loss_sum = 0
test_count = 0
# iterate through all users and try to make predictions based on trained RBM model
for id_user in range(nb_users):
    v0 = train_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    # can't make prediction for target test set if the test user did not rate any movie
    if len(vt[vt>=0]) > 0:
        _, h = rbm.sample_h(vt)
        _, v = rbm.sample_v(h)
        # compute the error
        test_loss_sum += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        test_count += 1
print("Finished testing with avg loss {}".format(test_loss_sum/test_count))
