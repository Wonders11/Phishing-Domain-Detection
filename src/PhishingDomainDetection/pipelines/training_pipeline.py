import os
import sys
import pandas as pd

from src.PhishingDomainDetection.logger import logging
from src.PhishingDomainDetection.exception import customexception
from src.PhishingDomainDetection.components.data_ingestion import DataIngestion

# creating data ingestion object
obj = DataIngestion()

obj.initiate_data_ingestion()
