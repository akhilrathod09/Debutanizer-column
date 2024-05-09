import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Models.LinearModel import *
from Models.utils import *

def main():
    #defining the parameters
    iterations=100
    learning_rate=0.01
    data=pd.read_csv("E:\CH512\Dataset\data.csv")
    #print(data)

    x_train,y_train,x_validate,y_validate,x_test,y_test=SplitDataset("E:\CH512\Dataset\data.csv",0.7,0.3,0)
    model=LinearRegression(iterations,learning_rate)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_validate)
    
    y_pred=y_pred.reshape(-1)
    error=model.error_log
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(error)), error, marker='o', linestyle='-')
    plt.title('Error Along the Length of Iteration for Linear Regression')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.savefig(os.path.join("Results", 'Linear_loss.png'))
    plt.grid(True)

    plt.show()
if __name__ == "__main__":
    main()