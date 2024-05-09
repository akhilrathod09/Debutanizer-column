import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Models.LinearModel import *
from Models.PolynomialModel import *
from Models.utils import *
from Models.Preprocessing.KFold import KFOLD
import os

def main():
    #defining the parameters
    iterations=100
    learning_rate=0.02
    data=pd.read_csv("E:\CH512\Dataset\data.csv")
    #print(data)

    x_train,y_train,x_validate,y_validate,x_test,y_test=SplitDataset("E:\CH512\Dataset\data.csv",0.7,0.3,0)
    model=PolynomialRegression(3,iterations,learning_rate)
    kfold=KFOLD(model)
    selected_features=kfold.ForwardSelection(x_train,y_train)
    print(selected_features)

    X_selected = x_train[:, selected_features]
    model.fit(X_selected,y_train)
    y_pred=model.predict(x_validate)
    
    y_pred=y_pred.reshape(-1)
    #result
    #MSE LOSS 0.029712149840376605
    #AIC 20.918184972209723
    #BIC 52.95347195554729
    aic=AIC(y_pred,y_validate,x_validate)
    bic=BIC(y_pred,y_validate,x_validate)
    loss=MSELoss(y_pred,y_validate)
    print("MSE LOSS",loss)
    print("AIC",aic)
    print("BIC",bic)
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_validate)), y_validate, color='blue',label='Real Values')
    plt.scatter(range(len(y_pred)), y_pred, color='red',label='Predicted Values')
    plt.title('Scatter Plot of Predicted vs. Real Values for KFOLD+POLYNOMIAL-ORDER-3')
    plt.ylabel('Real Values (y_validate)')
    plt.xlabel('Predicted Values (y_pred)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("Results", 'Kfold_Poly_3.png'))
    plt.show()

if __name__ == "__main__":
    main()