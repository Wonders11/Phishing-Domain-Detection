import pandas as pd
import numpy as np
import os
import sys
from src.PhishingDomainDetection.logger import logging
from src.PhishingDomainDetection.exception import customexception
from dataclasses import dataclass
from src.PhishingDomainDetection.utils.utils import save_object
from src.PhishingDomainDetection.utils.utils import find_best_model

from sklearn.linear_model import LogisticRegression
from sklearn import ensemble,svm,naive_bayes
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

        #     models={
        #     'LogisticRegression':LogisticRegression(),
        #     'RandomForest':ensemble.RandomForestClassifier(),
        #     'SVC':svm.SVC(),
        #     'NaiveBayes':naive_bayes.GaussianNB(),
        #     'DecisionTreeClassifier':DecisionTreeClassifier(),
        #     'XGBoost':xgb.XGBClassifier()
        # }

            
            
            # converting arrays to pandas dataframe
            X_train = pd.DataFrame(X_train)
            y_train = pd.DataFrame(y_train.ravel())
            X_test =  pd.DataFrame(X_test)
            y_test =  pd.DataFrame(y_test.ravel())

            param = find_best_model(X_train,y_train)

            logging.info('Hyperparameter tuning is completed and after comparing auc_roc_score and accuracy score of models we going to select model having hyperparameters: ' + str(param))

            args1={key: val for key, val in param.items() if key != 'classifier'}
            if param['classifier']=='LogReg':
                classifier_obj = LogisticRegression(**args1)
                model_name='LogisticRegression'
            elif param['classifier']=='RandomForest':
                classifier_obj = ensemble.RandomForestClassifier(**args1)
                model_name = 'RandomForest'
            elif param['classifier']=='SVC':
                classifier_obj = svm.SVC(**args1)
                model_name = 'SVC'
            elif param['classifier']=='NaiveBayes':
                classifier_obj = naive_bayes.GaussianNB(**args1)
                model_name = 'NaiveBayes'
            elif param['classifier']=='decision-tree':
                classifier_obj = DecisionTreeClassifier(**args1)
                model_name = 'DecisionTree'
            elif param['classifier']=='xgb':
                classifier_obj = xgb.XGBClassifier(**args1)
                model_name = 'XGBoost'

            classifier_obj.fit(X_train,y_train)
            logging.info('model trained Successfully!! Now testing begins')

            y_pred = classifier_obj.predict(X_test)
            logging.info('confusion_matrix ' + str(confusion_matrix(y_test, y_pred)))
            logging.info( 'accuracy_score ' + str(accuracy_score(y_test, y_pred)))
            logging.info( 'roc_auc_score ' + str(roc_auc_score(y_test, y_pred)))
            logging.info( 'classification_report ' + str(classification_report(y_test, y_pred)))
            logging.info('model tested successfully!!')

            logging.info('Starting to Save ML model')
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=classifier_obj
            )
            # save_model(
            #     file_path = self.model_trainer_config.trained_model_file_path,
            #     obj = classifier_obj,
            #     Model = model_name)
            logging.info('Model Saved')
            logging.info('Model Selection and tuning Completed')
            return model_name
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys) 