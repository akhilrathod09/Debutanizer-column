import numpy as np
import pandas as pd
from Models.utils import SplitDataset, is_numpy_array



class LassoRegression():
    def __init__(self,iterations,learning_rate,lamda):
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.lamda=lamda
        self.costStore=[]

    def fit(self,X,Y,use_dummy_lamda,dummy_lamda):
        if not is_numpy_array(X):
            raise AssertionError("X must be a NumPy array")
        
        if not is_numpy_array(Y):
            raise AssertionError("Y must be a NumPy array")
        
        if use_dummy_lamda:
            self.lamda = dummy_lamda

        #declaring the parameters required for stocastic gradient descent
        self.X_len,self.n_features=X.shape
        #print(X.shape)
        self.Weights=np.random.rand(self.n_features)
        self.bias=0
        self.X=X
        self.Y=Y
        for i in range(self.iterations):
            self.Gradient()

        return self
        
    def costFunction(self,y_pred,y_real):
        cost=(1/2*self.X_len)*np.sum(np.square(y_pred-y_real)) + self.lamda*np.sum(np.abs(self.Weights))
        return cost
    
    def Gradient(self):
        #self.Weights=self.Weights.reshape(-1,1)
        #print(self.Weights.shape)
        y_pred=self.predict(self.X)
        y_pred=y_pred.reshape(-1)
        cost=self.costFunction(y_pred,self.Y)
        self.costStore.append(cost)
        self.Weights = self.Weights - self.learning_rate * ((2 / self.X_len) * np.dot(self.X.T, (y_pred - self.Y)) + self.lamda * np.sign(self.Weights))
        self.bias=self.bias-(1/self.X_len)*(self.learning_rate)*np.sum(y_pred - self.Y)
        return self


    def predict(self,x):
        if not is_numpy_array(x):
            raise AssertionError("X must be a NumPy array")
        
        return x.dot(self.Weights)+self.bias
    
    def returnCostStore(self):
        return self.costStore
    
    def ImportanceFactor(self):
        return self.Weights