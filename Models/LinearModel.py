#Main model version 1.1

import numpy as np
import warnings
import os
import tqdm
from Models.utils import *

# X = np.array([[1, 5], [3, 4], [5, 6]])
# Y = np.array([7, 8, 9])
# X_with_intercept = np.c_[np.ones(X.shape[0]), X]
# print(X_with_intercept)

# B = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ Y
# print(B)

class LinearRegression():
    def __init__(self,iterations,learning_rate):
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.error_log=[]

    def fit(self,X,Y):
        if not is_numpy_array(X):
            raise AssertionError("X must be a NumPy array")
        
        if not is_numpy_array(Y):
            raise AssertionError("Y must be a NumPy array")
        
        #declaring the parameters required for stocastic gradient descent
        self.X_len,self.n_features=X.shape
        self.Weights=np.random.rand(self.n_features)
        self.bias=0
        self.X=X
        self.Y=Y
        for i in range(self.iterations):
            self.StochasticGradient()

        return self

    def StochasticGradient(self):
        self.Weights=self.Weights.reshape(-1,1)
        # print(self.X.shape)
        # print(self.Weights.shape)
        # print(self.Y.shape)
        y_pred=self.predict(self.X)
        y_pred=y_pred.reshape(-1)
        # print(self.Weights.shape)
        # print(y_pred.shape)
        # print(self.Y.shape)
        error=y_pred-self.Y
        mseloss=MSELoss(y_pred,self.Y)
        self.error_log.append(mseloss)

        
        #self.Weights[0]=self.Weights[0]-(1/self.X_len)*(self.learning_rate)*(np.matmul(self.X[:,0].transpose(),y_pred-self.Y))
        for j in range(self.n_features):
            self.Weights[j]=self.Weights[j]-(1/self.X_len)*(self.learning_rate)*(np.matmul(self.X[:,j].transpose(),y_pred-self.Y))

        self.bias=self.bias-(1/self.X_len)*(self.learning_rate)*np.sum(y_pred - self.Y)
        return self
            

    def predict(self,x):
        if not is_numpy_array(x):
            raise AssertionError("X must be a NumPy array")
        
        return x.dot(self.Weights)+self.bias


