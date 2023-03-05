from Insurance_Prediction import utils
from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.logger import logging
from Insurance_Prediction.entity import artifact_entity 
from Insurance_Prediction.entity import config_entity
import os,sys
from scipy import ks_2samp
import pandas as pd 
import numpy as np

class DataValidation:
    def __init__(self,data_validation_config:config_entity.DataValidationConfig):
        try:
            logging.info(f"{'>>'*20} Data Validation{'<<'*20}")
            self.data_validation_config=data_validation_config
        except Exception as e:
            raise InsuranceException(e,sys)

    def is_required_columns_exists(self,)->bool:
        pass

    

    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        pass

