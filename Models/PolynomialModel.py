#Main model version 1.1

import numpy as np
import warnings
import os
import tqdm
from Models.utils import *


class PolynomialRegression:

    def __init__(self, degree,iterations,learning_rate ):
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.error_log=[]

    def transform(self, X):
        X_transform = np.ones((X.shape[0], 1))
        for j in range(1, self.degree + 1): 
            x_pow = np.power(X, j)  
            X_transform = np.concatenate((X_transform, x_pow), axis=1)

        return X_transform

    def normalize(self, X):
        X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
        return X

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_len, self.n_features = self.X.shape

        #Transform X for polynomial features
        X_transform = self.transform(self.X)

        #Normalize X_transform
        X_normalize = self.normalize(X_transform)
        # Initialize weights
        self.Weights = np.zeros(X_transform.shape[1])

        # Gradient descent learning
        for i in range(self.iterations):
            self.StochasticGradient(X_normalize)

        return self
    
    def StochasticGradient(self,x_normal):
        y_pred=self.predict(self.X)
        y_pred=y_pred.reshape(-1)       
        error=y_pred-self.Y
        mseloss=MSELoss(y_pred,self.Y)
        self.error_log.append(mseloss)    
         
        self.Weights=self.Weights-(1/self.X_len)*(self.learning_rate)*(np.dot(x_normal.T,y_pred-self.Y))
        return self

    def predict(self, X):
        X_transform = self.transform(X)
        X_normalize = self.normalize(X_transform)
        return np.dot(X_normalize, self.Weights)
