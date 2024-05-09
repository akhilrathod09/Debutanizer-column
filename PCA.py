from Models.utils import SplitDataset
import matplotlib.pyplot as plt
import numpy as np
from Models.Preprocessing.PCA import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def PlotGraph(data):
    for column in data.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], bins=20, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join("Results", f'PCA-{column}.png'))
        plt.close()

        #
    plt.figure(figsize=(8, 6))
    sns.pairplot(data)
    plt.title('Pairwise Scatter Plots')
    plt.savefig(os.path.join("Results", 'PCA-scatter-Plot.png'))
    plt.close()

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


#plotting the graphs
PlotGraph(data_X)