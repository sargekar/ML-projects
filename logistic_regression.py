#!/usr/bin/env python
# coding: utf-8
# logistic_regression.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file contains a skeleton for implementing a simple version of the 
# Logistic Regression algorithm


import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle 
from sklearn.metrics import accuracy_score

M = np.genfromtxt('./data/monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytrn = M[:, 0]
Xtrn = M[:, 1:]

M = np.genfromtxt('./data/monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytst = M[:, 0]
Xtst = M[:, 1:]

class SimpleLogisiticRegression():
    """
    A simple Logisitc Regression Model which uses a fixed learning rate
    and Gradient Ascent to update the model weights
    """
    def __init__(self):
        self.w = []
        self.losses = []
        pass

        
    def initialize_weights(self, num_features):
        w = np.zeros((num_features))
        return w

    def compute_loss(self,  X, y):
        """
        Compute binary cross-entropy loss for given model weights, features, and label.
        :param w: model weights
        :param X: features
        :param y: label
        :return: loss   
        """
        m = len(y)
        h = self.sigmoid(X @ self.w)
        h = np.clip(h, 1e-15, 1 - 1e-15)  # Clip the predicted probabilities to avoid log(0)
        loss = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return loss

        raise Exception('Function not yet implemented!')

    
    def sigmoid(self, val):

        """
        Implement sigmoid function
        :param val: Input value (float or np.array)
        :return: sigmoid(Input value)
        """
        return 1 / (1 + np.exp(-val))
        # raise Exception('Function not yet implemented!')


    def gradient_ascent(self, w, X, y, lr):

        """
        Perform one step of gradient ascent to update current model weights. 
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        Update the model weights
        """
        m = X.shape[0]
        h = self.sigmoid(np.dot(X, self.w))
        gradient = np.dot(X.T, h - y) / m
        return lr * gradient
        # raise Exception('Function not yet implemented!')



    def fit(self,X, y, lr=0.1, iters=100, recompute=True):
        """
        Main training loop that takes initial model weights and updates them using gradient descent
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        :param recompute: Used to reinitialize weights to 0s. If false, it uses the existing weights Default True

        NOTE: Since we are using a single weight vector for gradient ascent and not using 
        a bias term we would need to append a column of 1's to the train set (X)

        """        
        if(recompute):
            #Reinitialize the model weights
            self.w = self.initialize_weights(X.shape[1])
            pass

        for _ in range(iters):
            loss = self.compute_loss(X, y)
            gradient = self.gradient_ascent(self.w, X, y,lr)
            self.w -=  gradient     # w -= lr  *gradient
            if _ % 100 == 0:
                self.losses.append(loss) 
            pass


    def predict_example(self, w, x):
        """
        Predicts the classification label for a single example x using the sigmoid function and model weights for a binary class example
        :param w: model weights
        :param x: example to predict
        :return: predicted label for x
        """
        h = self.sigmoid(np.dot(x, self.w))
        if h >= 0.5:
            return 1
        else:
            return 0
        # raise Exception('Function not yet implemented!')


    @staticmethod
    def compute_error(y_true, y_pred):
        """
        Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
        :param y_true: true label
        :param y_pred: predicted label
        :return: error rate = (1/n) * sum(y_true!=y_pred)
        """
        return np.mean(y_true != y_pred)
        # raise Exception('Function not yet implemented!')      
#          def compute_error_helper(self,y_true,y_pred):
#         return compute_error(y_true,y_pred)
        
ones_column = np.ones((122, 1))
ones_column_tst = np.ones((432, 1))


Xtrn = np.hstack((ones_column, Xtrn))
Xtst = np.hstack((ones_column_tst, Xtst))

def accuracy(train_pred,test_pred):
    print(np.sum(train_pred==test_pred)/len(test_pred))


train_errors = []
test_errors = []

for iter in [10, 100,1000,10000]:
    for a in [0.01,0.1, 0.33]:
        lr =  SimpleLogisiticRegression()
        lr.fit(Xtrn, ytrn, lr=a, iters=iter)
        train_pred = [lr.predict_example(lr.w, x) for x in Xtrn]
        train_error = lr.compute_error(ytrn, train_pred)
        train_errors.append(train_error)
        test_pred = [lr.predict_example(lr.w, x) for x in Xtst]
        test_error = lr.compute_error(ytst, test_pred)
        test_errors.append(test_error)
        accuracy(train_pred,ytrn)
        pass

#training errors
print(train_errors)

#testing errors
print(test_errors)

# best parameters is the one with least test_error
#which is the last one 
# 0.18287037037037038 
# #parameters - > iter = 10000 , lr = 0.33

accuracy(test_pred,ytst)

######### Part b:#########
# Retrain the models on the best parameters and save it as a pickle
# file in the format ‘NETID_lr.obj’.
iter = 10000
a = 0.33
lr =  SimpleLogisiticRegression()
lr.fit(Xtrn, ytrn, lr=a, iters=iter)
train_pred = [lr.predict_example(lr.w, x) for x in Xtrn]
train_error = lr.compute_error(ytrn, train_pred)
train_errors.append(train_error)
test_pred = [lr.predict_example(lr.w, x) for x in Xtst]
test_error = lr.compute_error(ytst, test_pred)
test_errors.append(test_error)
accuracy(train_pred,ytrn)
# best coefficients for the above model
print(lr.w)

netid = 'xxxxxxxxx'
file_pi = open('xxxxxxxxx_model_1.obj', 'wb')  #Use your NETID
pickle.dump(lr, file_pi)

######### Part c #########
# Compute the train and test errors using sklearn’s Logistic Regression Algorithm for comparision
clf = LogisticRegression(random_state=0).fit(Xtrn, ytrn)
clf.predict(Xtrn[:2, :])

#training error
1-clf.score(Xtrn, ytrn)

#testing error
1-clf.score(Xtst, ytst)

clf.predict_proba(Xtrn[:2, :])

######### Part d:#########
# For each of the learning rates, fit a Logistic Regression model that runs for 1000 iterations. 
# Store the training and testing loss at every 100th iteration and create three plots with epoch number on the x-axis and loss on the y-axis.
learning_rates = [0.01, 0.1, 0.33]
iters = 1000

train_loss = []
test_loss = []

for lr in learning_rates:
    # Create a new instance of the LogisticRegression class with the given learning rate and number of iterations
    model = SimpleLogisiticRegression()
    
    # Fit the model on the training data
    model.fit(Xtrn, ytrn,lr)
    train_loss_lr = [model.compute_loss(Xtrn, ytrn)]
    test_loss_lr = [model.compute_loss(Xtst, ytst)]
    for i in range(100, iters+1, 100):
        model.fit(Xtrn, ytrn, iters=100, recompute=False)
        train_loss_lr.append(model.compute_loss(Xtrn, ytrn))
        test_loss_lr.append(model.compute_loss(Xtst, ytst))
    train_loss.append(train_loss_lr)
    test_loss.append(test_loss_lr)
    
    
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, lr in enumerate(learning_rates):
    axs[i].plot(range(0, iters+1, 100), train_loss[i], label='Train Loss')
    axs[i].plot(range(0, iters+1, 100), test_loss[i], label='Test Loss')
    axs[i].set_title(f'Learning Rate: {lr}')
    axs[i].set_xlabel('Epochs')
    axs[i].set_ylabel('Loss')
    axs[i].legend()
plt.show()
