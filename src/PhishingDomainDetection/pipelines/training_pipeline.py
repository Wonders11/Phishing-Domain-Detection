# importing data ingestion class
from src.PhishingDomainDetection.components.data_ingestion import DataIngestion
from src.PhishingDomainDetection.components.data_transformation import DataTransformation
from src.PhishingDomainDetection.components.model_trainer import ModelTrainer


from src.PhishingDomainDetection.logger import logging
from src.PhishingDomainDetection.exception import customexception

# creation of data ingestion object
obj = DataIngestion() # once we create object of data ingestion, configuration object will also be created
# calling initiate_data_ingestion function
train_data_path, test_data_path = obj.initiate_data_ingestion()

# creation of data transformation object
data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path,test_data_path)

# creation of Model Trainer object
#model_trainer_obj = ModelTrainer()
#model_trainer_obj.initiate_model_training(train_arr,test_arr)