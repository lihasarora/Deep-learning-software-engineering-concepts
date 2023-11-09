# the objective of this script is to apply the data transformation and
# feature engineering steps
import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
# preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
import joblib
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # define constants
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
        self.numerical_columns = ["writing score","reading score"]
        self.categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
        self.target_column_name  = "math score"
    
    def get_data_transformer_object(self)->None:
        """Transforms the data based on data type

        Raises:
            CustomException: _description_

        Returns:
            Transformer Object
        """
        try:

            # pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy = "median")),
                    ("scalar" ,StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical columns Scaling completed")
            # categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy = "most_frequent")),
                    ("encoder" ,OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,self.numerical_columns),
                    ("cat_pipeline",cat_pipeline,self.categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Train and test data loaded")
            logging.info("Obtain Preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            input_feature_train_df = train_df.drop(columns = self.target_column_name, axis = 1)
            target_feature_train_df = train_df[self.target_column_name]
            
            
            input_feature_test_df = test_df.drop(columns = self.target_column_name, axis = 1)
            target_feature_test_df = test_df[self.target_column_name]
            
            logging.info("Apply preprocessing on training and test dataframe")
            logging.info("preprocessor on the train data")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info("preprocessor on the test data")
            input_feature_test_arr  = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("saving the train and test data in the form of numpy array")
            logging.info("saving preprocessing object")
            
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,preprocessing_obj
                )
            
            return (
                train_arr, 
                test_path,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
        

if __name__ == '__main__':
    obj = DataTransformation()
    prepro = obj.get_data_transformer_object()
    