
from Models.utils import *
import matplotlib.pyplot as plt
import numpy as np
from Models.Preprocessing.PCA import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Models.LinearModel import LinearRegression
from Models.NeuralNet import neuralNetworkMultiHidden
from Models.SimpleNet import neuralNetworkSingleHidden

def plot_residuals_single(residuals):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(residuals)), residuals, marker='o', linestyle='-', color='blue')
    plt.title('Residuals vs. Number of Samples for Single Hidden layer')
    plt.xlabel('Number of Samples')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()

def plot_residuals_multiple(residuals):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(residuals)), residuals, marker='o', linestyle='-', color='blue')
    plt.title('Residuals vs. Number of Samples for Multi Hidden layer')
    plt.xlabel('Number of Samples')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()

def autocorrelation_Single(residuals):
    residuals_flat = np.array(residuals).flatten()
    
    autocorr = np.correlate(residuals_flat, residuals_flat, mode='full')
    autocorr = autocorr / np.max(autocorr) 
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(autocorr)), autocorr, marker='o', linestyle='-', color='red')
    plt.title('Autocorrelation Coefficient w.r.t. Residuals single Hidden layer')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation Coefficient')
    plt.grid(True)
    plt.show()

def autocorrelation_multiple(residuals):
    residuals_flat = np.array(residuals).flatten()
    
    autocorr = np.correlate(residuals_flat, residuals_flat, mode='full')
    autocorr = autocorr / np.max(autocorr) 
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(autocorr)), autocorr, marker='o', linestyle='-', color='red')
    plt.title('Autocorrelation Coefficient w.r.t. Residuals Multi Hidden layer')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation Coefficient')
    plt.grid(True)
    plt.show()


iterations=100
learning_rate=0.02
X,y,x_validate,y_validate,x_test,y_test=SplitDataset("E:\CH512\Dataset\data.csv",0.7,0.3,0)
data=pd.read_csv("E:\CH512\Dataset\data.csv")
data_X=data.drop("y", axis=1)
print(data_X.columns)
pca=PCA(2)
pca.fit(X)

print('Components:\n', pca.filtered_components)
print('Explained variance ratio:\n', pca.explained_variance_ratio)

cum_explained_variance = np.cumsum(pca.explained_variance_ratio)
print('Cumulative explained variance:\n', cum_explained_variance)

X_pca = pca.transformData(X) # Apply dimensionality reduction to X.
print('Transformed data shape:', X_pca.shape)

x_test_pca=pca.transformData(x_validate)



##declaring the model here
print(X_pca.shape[1])
layer_sizes = [X_pca.shape[1], 3, 1]  # Input layer, Hidden layer 1, Output layer
model1 = neuralNetworkSingleHidden(X_pca.shape[1],4,1,0.001)
model2 = neuralNetworkMultiHidden(X_pca.shape[1],4,4,1,0.01)

# Train the neural network
model1.train(X_pca, y)
residual1=model1.residuals

y_pred_1=model1.predict(x_test_pca)
y_pred_1=y_pred_1.reshape(-1)

model2.train(X_pca, y)
residual2=model2.residuals

y_pred_2=model2.predict(x_test_pca)

    
y_pred_2=y_pred_2.reshape(-1)
print(np.shape(residual1))
plot_residuals_single(residual1)
plot_residuals_multiple(residual2)

autocorrelation_Single(residual1)
autocorrelation_multiple(residual2)
# plt.figure(figsize=(8, 6))
# plt.scatter(range(len(y_validate)), y_validate, color='blue',label='Real Values')
# plt.scatter(range(len(y_pred)), y_pred, color='red',label='Predicted Values')
# plt.title('Scatter Plot of Predicted vs. Real Values for PCA+Singlelayer')
# plt.ylabel('Real Values (y_validate)')
# plt.xlabel('Predicted Values (y_pred)')
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join("Results", 'PCA_NN_Single.png'))
# plt.show()


