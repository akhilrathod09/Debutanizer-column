from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from Models.SimpleNet import neuralNetworkSingleHidden
from Models.utils import *
import matplotlib.pyplot as plt
import numpy as np
from Models.Preprocessing.PCA import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import numpy as np

class NearestNeighbors:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        
    def fit(self, X):
        self.X_train = X
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def kneighbors(self, X_test):
        distances = []
        indices = []
        for x_test in X_test:
            dist = [self.euclidean_distance(x_test, x_train) for x_train in self.X_train]
            sorted_indices = np.argsort(dist)
            #distances.append(dist[sorted_indices[:self.n_neighbors]])
            indices.append(sorted_indices[:self.n_neighbors])
        return np.array(indices)


def split_data(X, y, test_size=0.3):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val

def build_database(X_train, y_train):
    return X_train, y_train

def knn_search(X_train, y_train, X_val, k=5):
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X_train)
    indices = knn.kneighbors(X_val)
    return indices, y_train[indices]

def train_local_model(X, y):
    model = neuralNetworkSingleHidden(X.shape[1],4,1,0.001)
    model.train(X, y)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return mse


X_train,y_train,X_val,y_val,x_test,y_test=SplitDataset("E:\CH512\Dataset\data.csv",0.7,0.3,0)


#Building a database with training data
database_X, database_y = build_database(X_train, y_train)

#Performing k-NN search for validation data
indices, knn_y_train = knn_search(database_X, database_y, X_val, k=5)

#Training local model using k-NN search results
local_models = []
for i in range(len(X_val)):
    local_X = X_train[indices[i]]
    local_y = knn_y_train[i]
    local_model = train_local_model(local_X, local_y)
    local_models.append(local_model)

#Evaluating model for each validation data point
total_mse = 0
for i in range(len(X_val)):
    mse = evaluate_model(local_models[i], X_val[i].reshape(1, -1), y_val[i].reshape(1, -1))
    total_mse += mse
average_mse = total_mse / len(X_val)

print(average_mse)