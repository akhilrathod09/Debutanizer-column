import numpy as np
import pandas as pd
from Models.utils import SplitDataset,MSELoss


class KFOLD():
    def __init__(self,model):
        self.n_features=None
        self.model=model
        self.selected_features=[]

    def ForwardSelection(self,X,Y):
        self.n_features=X.shape[1]
        
        for i in range(self.n_features):
            best_feature_index=-1
            Best_Mse=np.inf

            for j in range (self.n_features):
                if j not in self.selected_features:
                    curr_features=self.selected_features+[j]
                    selected_subset_features=X[:,curr_features]

                    #using the initiated model
                    self.model.fit(selected_subset_features,Y)
                    y_pred=self.model.predict(selected_subset_features)

                    loss=MSELoss(y_pred,Y)
                    if Best_Mse>loss:
                        Best_Mse=loss
                        best_feature_index=j
            self.selected_features.append(best_feature_index)
            print("Step", i + 1, "- Selected feature:", best_feature_index, "- MSE:", Best_Mse)

        return self.selected_features
    


            