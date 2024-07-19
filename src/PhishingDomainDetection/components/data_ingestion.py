import pandas as pd
import numpy as np
from src.PhishingDomainDetection.logger import logging
from sklearn.model_selection import train_test_split
from src.PhishingDomainDetection.exception import customexception
from dataclasses import dataclass
from pathlib import Path
import os
import sys

# data class
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # logging info
        logging.info("Data Ingestion started")

        try:
            data = pd.read_csv(Path(os.path.join("notebooks/data","dataset_full.csv"))) # system independent path -works well on windows, linux
            logging.info("Data readed successfully")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Saved the raw data in artifacts folder")

            logging.info("Performing test train split")
            train_data,test_data = train_test_split(data,test_size=0.25)
            logging.info("Data splitting completed")

            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Saved the test and train data in artifacts folder")

            logging.info("Data Ingestion Completed")

        except Exception as e:
            logging.info("Exception occured during Data Ingestion Stage")
            raise customexception(e,sys)