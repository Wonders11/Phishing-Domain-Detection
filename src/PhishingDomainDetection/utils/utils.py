import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.PhishingDomainDetection.logger import logging
from src.PhishingDomainDetection.exception import customexception
from pathlib import Path
import optuna
from sklearn import linear_model
from sklearn import ensemble
import sklearn.svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
import shutil


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)

"""   
def save_model(file_path,model,filename):
    model_directory = 'models/'
    try:
        path = os.path.join(model_directory,filename) #create seperate directory for each cluster
        if os.path.isdir(path): #remove previously existing models for each clusters
            shutil.rmtree(model_directory)
            os.makedirs(path)
        else:
            os.makedirs(path) #
            with open(path +'/' + filename+'.pkl','wb') as f:
                pickle.dump(model, f) # save the model to file

    except Exception as e:
        raise e
"""

def save_model(file_path, model, filename):
    """
    Saves the model file to the specified directory.

    Args:
        file_path (str): The path to the directory where the model will be saved.
        model: The model object to be saved.
        filename (str): The filename for the saved model.
    """

    try:
        model_path = os.path.join(file_path, filename + '.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        raise e


def objective(trial,X_train,y_train):
    """
    Description: This method is objective function for optuna.
    Output: returns roc_auc score and accuracy score of best model based on training data
    """
    try:
        classifier_name = trial.suggest_categorical("classifier", ["LogReg", "RandomForest", "SVC", "NaiveBayes",
                                                                   "decision-tree", "xgb"])

        # Step 2. Setup values for the hyperparameters:
        if classifier_name == 'LogReg':
            C = trial.suggest_uniform('C', 0.01, 10)
            classifier_obj = linear_model.LogisticRegression(C=C)

        elif classifier_name == 'RandomForest':
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 20)
            classifier_obj = sklearn.ensemble.RandomForestClassifier(min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)

        elif classifier_name == 'SVC':
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            classifier_obj = sklearn.svm.SVC(kernel=kernel)

        elif classifier_name == 'NaiveBayes':
            var_smoothing = trial.suggest_float("var_smoothing", 1e-4, 0.3, log=True)
            classifier_obj = sklearn.naive_bayes.GaussianNB(var_smoothing=var_smoothing)

        elif classifier_name == 'decision-tree':
            max_depth = trial.suggest_int('max_depth', 5, X_train.shape[1])
            classifier_obj = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth)

        elif classifier_name == 'xgb':
            alpha = trial.suggest_float('alpha', 1e-4, 1)
            subsample = trial.suggest_float('subsample', .1, .5)
            classifier_obj = xgb.XGBClassifier(alpha=alpha, subsample=subsample)

        #Step 3: Scoring method:
        accuracy = []
        roc_auc =[]
        skf = StratifiedKFold(n_splits=10, random_state=None)
        skf.get_n_splits(X_train, y_train)

        # X is the feature set and y is the target
        for train_index, test_index in skf.split(X_train, y_train):
            X1_train, X1_test = X_train.iloc[train_index], X_train.iloc[test_index]
            y1_train, y1_test = y_train.iloc[train_index], y_train.iloc[test_index]

            classifier_obj.fit(X1_train, y1_train)
            prediction = classifier_obj.predict(X1_test)
            score = accuracy_score(prediction, y1_test)
            accuracy.append(score)
            try:
                score1 = roc_auc_score(prediction, y1_test)
                roc_auc.append(score1)
            except ValueError:
                pass
        accuracy=np.array(accuracy).mean()
        roc_auc = np.array(roc_auc).mean()

        return accuracy, roc_auc
    except Exception as e:
        raise e

def find_best_model(X_train,y_train):
    """
    Description: This method finds the best model based on accuracy and roc_auc_score.
    Output: Return the best model hyperparameters 
    """

    try:
        sampler = optuna.samplers.NSGAIISampler()
        func = lambda trial: objective(trial, X_train,y_train)
        study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
        study.optimize(func, n_trials=10)
        trial = study.best_trials
        param = trial[0].params
        return param

    except Exception as e:
        raise e

    
# def evaluate_model(X_train,y_train,X_test,y_test,models):
#     try:
#         report = {}
#         for i in range(len(models)):
#             model = list(models.values())[i]
#             # Train model
#             model.fit(X_train,y_train)

#             # Predict Testing data
#             y_test_pred =model.predict(X_test)

#             # Get R2 scores for train and test data
#             #train_model_score = r2_score(ytrain,y_train_pred)
#             test_model_score = r2_score(y_test,y_test_pred)

#             report[list(models.keys())[i]] =  test_model_score

#         return report

    # except Exception as e:
    #     logging.info('Exception occured during model training')
    #     raise customexception(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)