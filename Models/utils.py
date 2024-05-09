#helper functions version 1.1
import pandas as pd
import numpy as np


#following function splits the dataset with the given ratio like 0.8:0.1:0.1
def SplitDataset(csv_file,train_ratio=0.7,validation_ratio=0.15,test_ratio=0.15):
    """
    Splits the data with the given ratio
    
    Parameters:
    - csv_file: csv location
    - train_ratio: train_ratio
    - validation_ratio: validation_ratio
    - test_ratio: test_ratio
    
    Returns:
    - 
    """
    #loading the dataset
    data=pd.read_csv(csv_file)

    #shuffling the dataframe
    data=data.sample(frac=1).reset_index(drop=True)

    total_index=len(data)
    train_index=int(train_ratio*total_index)
    val_index=int(validation_ratio*total_index)

    #splitting the dataset
    train_data=data[:train_index]
    validation_data=data[train_index:train_index+val_index]
    test_data=data[train_index+val_index:]

    #adjusting according to the data
    x_train = train_data.drop("y", axis=1).values
    y_train=train_data["y"].values

    x_validate=validation_data.drop("y", axis=1).values
    y_validate=validation_data["y"].values

    x_test=test_data.drop("y", axis=1).values
    y_test=test_data["y"].values

    return x_train,y_train,x_validate,y_validate,x_test,y_test

def SplitDataset_DataFrame(dataframe,train_ratio=0.7,validation_ratio=0.15,test_ratio=0.15):
    """
    Splits the data with the given ratio
    
    Parameters:
    - csv_file: csv location
    - train_ratio: train_ratio
    - validation_ratio: validation_ratio
    - test_ratio: test_ratio
    
    Returns:
    - 
    """
    #loading the dataset
    data=dataframe

    #shuffling the dataframe
    data=data.sample(frac=1).reset_index(drop=True)

    total_index=len(data)
    train_index=int(train_ratio*total_index)
    val_index=int(validation_ratio*total_index)

    #splitting the dataset
    train_data=data[:train_index]
    validation_data=data[train_index:train_index+val_index]
    test_data=data[train_index+val_index:]

    #adjusting according to the data
    x_train = train_data.drop("y", axis=1).values
    y_train=train_data["y"].values

    x_validate=validation_data.drop("y", axis=1).values
    y_validate=validation_data["y"].values

    x_test=test_data.drop("y", axis=1).values
    y_test=test_data["y"].values

    return x_train,y_train,x_validate,y_validate,x_test,y_test




def euclidean_distance(array1, array2):
    """
    Calculate the Euclidean distance between two arrays.
    
    Parameters:
    - array1: First array
    - array2: Second array
    
    Returns:
    - Euclidean distance
    """
    distance = np.linalg.norm(array1 - array2, axis=1)
    return distance


def is_numpy_array(obj):
    """
    To check if the given object is a NumPy array.

    Parameters:
    - obj: Object to check

    Returns:
    - Bool
    """
    return isinstance(obj, np.ndarray)

#LOSS functions are defined here

def MSELoss(y_pred,y_real):
    if len(y_pred)!=len(y_real):
        raise AssertionError("Both arrays have to of same length")
        
    sq_err=(y_real-y_pred)**2
    loss=np.mean(sq_err)
    return loss


def SSE(y_pred, y_real):
    if len(y_pred)!=len(y_real):
        raise AssertionError("Both arrays have to of same length")
    
    errors = y_pred - y_real
    squared_errors = np.square(errors)
    
    sse = np.sum(squared_errors)
    
    return sse

def AIC(y_pred,y_real,x):
    mseloss=MSELoss(y_pred,y_real)
    x_len=y_pred.shape[0]
    n_features=x.shape[1]
    
    aic=2*n_features-2*np.log(mseloss) 

    return aic

def BIC(y_pred,y_real,x):
    mseloss=MSELoss(y_pred,y_real)
    x_len=y_pred.shape[0]
    n_features=x.shape[1]

    bic=np.log(x_len)*n_features - 2*np.log(mseloss)

    return bic

def r_squared(y_real, y_pred):
    actual_mean = np.mean(y_real)
    total_sum_of_squares = np.sum((y_real - actual_mean) ** 2)
    residual_sum_of_squares = np.sum((y_real - y_pred) ** 2)
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2
