# here we will be training different models and check the performance

import os
import sys
from typing import Optional
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
# Evaluation Metrics
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import *
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

@dataclass
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path = None):
        
        try:
            logging.info("Extract the Y variable from the train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
                )
           
            logging.info("Start model training")
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression()
            }
            
            # Explicitly annotating the variable with the variable type clarifies things a little bit.
            # This may not be typically helpful for  a straight forward variable
            logging.info("Executing the evaluate model function")
            
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            model_report:dict = evaluate_model(
                X_train = X_train,
                y_train = y_train,
                X_test  = X_test,
                y_test  = y_test,
                models  = models
            )            
            # get the best model score from the dictionary
            model_name  = np.array(list(model_report.keys()))
            model_score = np.array(list(model_report.values()))
            
            logging.info("Selecting the best model")
            # print(model_score,model_report)
            best_model_score = np.max(model_score)
            # print(f"\n{np.argmax(model_score)}")
            best_model_name  = model_name[np.argmax(model_score)]
            print(best_model_name)
            # print(best_model_score)
            
            
            best_model = models[best_model_name]
            
            if best_model_score<0.2:
                raise CustomException("No best model found")
            
            logging.info("Saving the model as a pkl file")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj = best_model
            )
            
            logging.info("Predicting using the best model")
            predicted = best_model.predict(X_test)
            
            best_r2_score = r2_score(y_true=y_test, y_pred=predicted)

            return best_r2_score
            
        except Exception as e:
            raise CustomException(e,sys)
    


if __name__ == '__main__':
    X_test = np.random.random_integers(low = 1,high = 100, size = 100).reshape((20,5))
    X_test = np.random.random_integers(low = 1,high = 100, size = 100).reshape((20,5))
    
    pass