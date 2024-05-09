import numpy as np
import pandas as pd
from Models.utils import SplitDataset, is_numpy_array
import matplotlib.pyplot as plt
import seaborn as sns

class PCA():
    def __init__(self,n_features):
        self.n_features=n_features
    
    def fit(self,X):
        self.mean=None
        self.filtered_components=None
        self.scale=None
        self.X=X
        #getting the mean and scale
        self.mean=np.mean(X,axis=0)
        self.scale=np.std(X,axis=0)

        #step 1
        #centering the data
        X_standard_deviation=(X-self.mean)/self.scale


        #step 2
        #getting the covariance matrix 
        cov_matrix=np.cov(X_standard_deviation.T)

        #step 3
        #getting the eigen values and eigen vectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        #step 4
        #ordering the eigenvectors in order and getting the important features
        max_index=np.argmax(np.abs(eigenvectors),axis=0)
        eigenvectors = eigenvectors.T
       
        eigenPairs = [(np.abs(eigenvalues[i]), eigenvectors[i,:]) for i in range(len(eigenvalues))]
        eigenPairs.sort(key=lambda x: x[0], reverse=True)
        eigenvalues = np.array([x[0] for x in eigenPairs])
        eigenvectors = np.array([x[1] for x in eigenPairs])
        # eigenvalues = eigenvalues[max_index]
        # eigenvectors = eigenvectors[max_index]

        #step 5 
        #storing the first n_features eigenvectors
        self.filtered_components = eigenvectors[:self.n_features,:]
        self.explained_variance_ratio = [i/np.sum(eigenvalues) for i in eigenvalues[:self.n_features]]
        
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)


    def transformData(self,X):
        X=X-self.mean
        return np.dot(X,self.filtered_components.T)
    
    def PlotGraph(self,data):
        for column in data.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data[column], bins=20, kde=True)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

        #
        sns.pairplot(data)
        plt.title('Pairwise Scatter Plots')
        plt.show()

    

