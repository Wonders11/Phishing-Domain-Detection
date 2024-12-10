import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.PhishingDomainDetection.exception import customexception
from src.PhishingDomainDetection.logger import logging

from sklearn.compose import ColumnTransformer # for creating pipelines
# from sklearn.impute import SimpleImputer # not required as there are no missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # very imp
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

from src.PhishingDomainDetection.utils.utils import save_object # for saving object(transformed data) file generated 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

# Handling Outliers in train_df and test_df
def OutlierHandler(df):
    for feature in df.columns:
        IQR = df[feature].quantile(0.75) - df[feature].quantile(0.25)

        # calculation of upper and lower boundary value
        lower_bridge = df[feature].quantile(0.25) - (IQR * 1.5)
        upper_bridge = df[feature].quantile(0.75) + (IQR * 1.5)

        df.loc[df[feature] < lower_bridge, feature] = lower_bridge
        df.loc[df[feature] >= upper_bridge, feature] = upper_bridge

    return df

# Feature Selection based on Correlation
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
 
    # this method is responsible for creation of .pkl
    def get_data_transformation(self,temp_df):
        try:
            # Getting numerical and categorical columns
            # Observation: There are no categorical columns and all other columns are numerical

            # here main code for data transformation
            numerical_columns = temp_df.columns[temp_df.dtypes!="object"]

            # Pipeline for numerical data transformation
            num_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler())
            ])
            
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

            # temp_df = train_df # for accessing numerical columns at later stage

            logging.info("Completed reading train and test data")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string}')

            # Splitting features and target
            X_train = train_df.drop("phishing", axis=1)
            y_train = train_df["phishing"]
            X_test = test_df.drop("phishing", axis=1)
            y_test = test_df["phishing"]

            logging.info("Data Transformation initiated")

            # Handling imbalanced data using SMOTE
            # Synthetic Minority Over-sampling Technique ,Resampling the minority class.
            logging.info("Handling imbalanced data using SMOTE")
            sm = SMOTE(sampling_strategy='minority', random_state=45) 

            # Fit the model to generate the data.
            oversampled_X_train, oversampled_Y_train = sm.fit_resample(X_train, y_train)
            # train_data = pd.concat([pd.DataFrame(oversampled_Y_train), pd.DataFrame(oversampled_X_train)], axis=1)
            train_data = pd.concat([pd.DataFrame(oversampled_Y_train).reset_index(drop=True), 
                        pd.DataFrame(oversampled_X_train).reset_index(drop=True)], axis=1)

            oversampled_X_test, oversampled_Y_test = sm.fit_resample(X_test, y_test)
            # test_data = pd.concat([pd.DataFrame(oversampled_Y_test), pd.DataFrame(oversampled_X_test)], axis=1)
            test_data = pd.concat([pd.DataFrame(oversampled_Y_test).reset_index(drop=True), 
                        pd.DataFrame(oversampled_X_test).reset_index(drop=True)], axis=1)
            logging.info("Imbalanced Dataset handled by SMOTE")

            # Here we will get same balanced dataset that means same number of data for 2 class

            # Handling outliers in training dataset
            logging.info("Handling outliers in all features of training dataset")
            train_data = OutlierHandler(train_data)
            logging.info("Outliers have been handled in training data")

            # Handling outliers in testing dataset
            logging.info("Handling outliers in all features of testing dataset")
            test_data = OutlierHandler(test_data)
            logging.info("Outliers have been handled in testing data")

            # Splitting features and target
            X_train = train_data.drop("phishing", axis=1)
            y_train = train_data["phishing"]
            X_test = test_data.drop("phishing", axis=1)
            y_test = test_data["phishing"]
            
            logging.info("Feature selection started")
            #Feature selection:Finding correlated features and removing those features which are 85% correlated in training data
            corr_features = correlation(X_train, 0.85)
            #Removing correlated features from training data
            X_train.drop(corr_features,axis=1,inplace=True)
            logging.info("Removed correlated features from training data")

            # Feature selection:Finding correlated features and removing those features which are 85% correlated in test data
            # Removing correlated features from test data (same features to be removed from X_test as removed from X_train data)
            X_test.drop(corr_features,axis=1,inplace=True)
            logging.info("Removed correlated features from test data")

            # Finding features having 0 varience in  training data
            var_thres = VarianceThreshold(threshold=0)
            var_thres.fit(X_train)
            constant_columns = [column for column in X_train.columns
                            if column not in X_train.columns[var_thres.get_support()]]

            #dropping features having 0 varience from train data
            X_train.drop(constant_columns, axis=1, inplace=True)
            logging.info("Removed features having 0 varience from training data")

            # dropping features having 0 varience from test data (same features as dropped from train_data) 
            X_test.drop(constant_columns,axis=1,inplace=True)
            logging.info("Removed features having 0 varience from test data")

            #creating new test and train dataframe
            df_final_train = pd.DataFrame(X_train)
            df_final_test = pd.DataFrame(X_test)

            logging.info(f'df_final_train Dataframe Head : \n{df_final_train.head().to_string}')
            logging.info(f'df_final_test Dataframe Head : \n{df_final_test.head().to_string}')

            df_final_train.to_csv("artifacts/df_final_train.csv")
            df_final_test.to_csv("artifacts/df_final_test.csv")
            
            preprocessing_obj = self.get_data_transformation(df_final_train) # df_final_train is passed just to extract information about features
            
            numerical_columns = df_final_train.columns[df_final_train.dtypes!="object"] # name of the columns

            logging.info("Applying preprocessor object on training and testing data")
            X_train_transformed = preprocessing_obj.fit_transform(df_final_train)
            pd.DataFrame(X_train_transformed,columns=numerical_columns).to_csv("artifacts/X_train_transformed.csv",index=False)

            X_test_transformed = preprocessing_obj.transform(df_final_test)
            pd.DataFrame(X_test_transformed,columns=numerical_columns).to_csv("artifacts/X_test_transformed.csv",index=False)
            logging.info("Applied preprocessing on training and testing data")

            # Combine the features and target back into arrays
            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            # creating dataframe and storing it in artifacts folder
            final_train_data = pd.DataFrame(train_arr)
            final_test_data = pd.DataFrame(test_arr)
            final_train_data.to_csv("artifacts/final_train_data.csv")
            final_test_data.to_csv("artifacts/final_test_data.csv")
            logging.info("Saved final train and test dataframe to artifacts folder")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor pickle file saved")

            return train_arr, test_arr

        except Exception as e:
            logging.info("Exception occured during initialize_data_transformation")
            raise customexception(e,sys)