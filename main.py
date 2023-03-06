from Insurance_Prediction.logger import logging
from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.utils import get_collection_as_dataframe
import os,sys
from Insurance_Prediction.entity import config_entity
from Insurance_Prediction.entity.config_entity import DataIngestionConfig
from Insurance_Prediction.components.data_ingestion import DataIngestion
from Insurance_Prediction.components.data_validation import DataValidation
if __name__=="__main__":
    try:
        training_pipeline_config=config_entity.TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()

        data_validation_config=config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation=DataValidation(data_validation_config=data_validation_config,
                        data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact=data_validation.initiate_data_validation()
    except Exception as e:
        print(e)