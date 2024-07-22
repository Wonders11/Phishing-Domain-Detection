import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.PhishingDomainDetection.exception import customexception
from src.PhishingDomainDetection.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.PhishingDomainDetection.utils.utils import save_object # for saving object(transformed data) file generated 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self,train_df):
        try:
            logging.info("Data Transformation initiated")

            # here main code for data transformation   
            # There is no categorical data

            numerical_columns = train_df.columns[train_df.dtypes!="object"]

            # creating numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median'))
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns)
            ])

            return preprocessor
                
        except Exception as e:
            logging.info("Exception occured during get_data_transformation")
            raise customexception(e,sys)

    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)      
            test_df = pd.read_csv(test_path)

            logging.info("Completed reading train and test data")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string}')

            preprocessing_obj = self.get_data_transformation(train_df)

            # segregating columns
            target_column_name = 'phishing'
            drop_columns = [target_column_name]

            # segregating independent and dependent feature
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            # transforming train data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(target_feature_train_df)
            logging.info("Applied preprocessing object on training and testing datasets")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path
                obj = preprocessing_obj
            )

        except Exception as e:
            logging.info("Exception occured during initialize_data_transformation")
            raise customexception(e,sys)