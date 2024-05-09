import numpy as np
import pandas as pd
from Models.utils import SplitDataset, is_numpy_array,MSELoss

class GridsearchCV():
    def __init__(self,model,params):
        self.model=model
        self.params=params

    def Search(self,x_train,y_train,x_validate,y_validate):
        if not is_numpy_array(x_train):
            raise AssertionError("X must be a NumPy array")
        
        if not is_numpy_array(y_train):
            raise AssertionError("Y must be a NumPy array")
        
        if not is_numpy_array(x_validate):
            raise AssertionError("X must be a NumPy array")
        
        if not is_numpy_array(y_validate):
            raise AssertionError("Y must be a NumPy array")
        
        max_accuracy=np.inf
        best_param=0

        print("Finding the best parameter among :", self.params)

        for i in range (len(self.params)):
            self.model.fit(x_train,y_train,use_dummy_lamda=True,dummy_lamda=self.params[i])

            y_pred=self.model.predict(x_validate)
            current_loss=MSELoss(y_pred,y_validate)
            if max_accuracy>current_loss:
                max_accuracy=current_loss
                best_param=self.params[i]

        print("minimum MSE LOSS acheived is :",max_accuracy)
        print("Best Param is: ",best_param)

        return max_accuracy, best_param
    

            
        


    