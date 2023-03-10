from Insurance_Prediction.logger import logging
from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.utils import get_collection_as_dataframe
import os,sys
from Insurance_Prediction.entity import config_entity
from Insurance_Prediction.entity.config_entity import DataIngestionConfig
from Insurance_Prediction.components.data_ingestion import DataIngestion
from Insurance_Prediction.components.data_validation import DataValidation
from Insurance_Prediction.components.data_transformation import DataTransformation
from Insurance_Prediction.components.model_trainer import ModelTrainer
if __name__=="__main__":
    try:
        #Data training phase 
        training_pipeline_config=config_entity.TrainingPipelineConfig()

        #Data ingestiondone 
        data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()

        #data validation done 
        data_validation_config=config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation=DataValidation(data_validation_config=data_validation_config,
                        data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact=data_validation.initiate_data_validation()

        #data transformation done 
        data_transformation_config=config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation=DataTransformation(data_transformation_config=data_transformation_config,
                                     data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact=data_transformation.initiate_data_transformation()

        #modeltrianer
        model_trainer_config=config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config, 
                                    data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()   
    except Exception as e:
        print(e)