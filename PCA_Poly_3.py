from Models.utils import *
import matplotlib.pyplot as plt
import numpy as np
from Models.Preprocessing.PCA import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Models.LinearModel import LinearRegression
from Models.PolynomialModel import *

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

model=PolynomialRegression(3,iterations,learning_rate)
model.fit(X_pca,y)
y_pred=model.predict(X_pca)
    
y_pred=y_pred.reshape(-1)
    #result
# MSE LOSS 0.024285145721552204
# AIC 11.435780805453478
# BIC 22.28291769397181
aic=AIC(y_pred,y,X_pca)
bic=BIC(y_pred,y,X_pca)
loss=MSELoss(y_pred,y)
print("MSE LOSS",loss)
print("AIC",aic)
print("BIC",bic)
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y)), y, color='blue',label='Real Values')
plt.scatter(range(len(y_pred)), y_pred, color='red',label='Predicted Values')
plt.title('Scatter Plot of Predicted vs. Real Values for PCA+POLY-3')
plt.ylabel('Real Values (y_validate)')
plt.xlabel('Predicted Values (y_pred)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join("Results", 'PCA_Poly_3.png'))
plt.show()

