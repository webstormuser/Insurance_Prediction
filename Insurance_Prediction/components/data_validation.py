from Insurance_Prediction import utils
from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.logger import logging
from Insurance_Prediction.entity import artifact_entity 
from Insurance_Prediction.entity import config_entity
import os,sys
from scipy.stats import ks_2samp
import pandas as pd 
import numpy as np
from Insurance_Prediction import utils

class DataValidation:
    def __init__(self,data_validation_config:config_entity.DataValidationConfig,
                        data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation{'<<'*20}")
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.validation_error=dict()
        except Exception as e:
            raise InsuranceException(e,sys)

    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
            
            base_columns=base_df.columns
            current_columns=current_df.columns
            missing_columns=[]
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"column [{base} is not available]")
                    missing_columns.append(base_column)
            if len(missing_columns)>0:
                self.validation_error[report_key_name]=missing_columns
        except Exception as e:
            raise InsuranceException(e ,sys)
    
    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report=dict()
            base_columns=base_df.columns
            current_columns=current_df.columns
            #Null hypothesis is that base column data are drawn from same distribution 
            for base_column in base_columns:
                base_data,current_data=base_df[base_column],current_df[base_column]
                same_distribution=ks_2samp(base_data,current_data)
                if same_distribution.pvalue>0.05:
                    drift_report[base_column]={
                        "pvalues":same_distribution.pvalue,
                        "same_distribution":True
                    }
                    #We accept null hypothesis  
                else:
                    drift_report[base_column]={
                        "pvalues":same_distribution.pvalue,
                        "same_distribution":False
                    }
                    #different distribution
                self.validation_error[report_key_name]=drift_report
        except Exception as e:
                raise InsuranceException(e,sys)

    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading base dataframe")
            base_df=pd.read_csv(self.data_validation_config.base_file_path)
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            train_df_column_status=self.is_required_columns_exists(base_df=base_df,current_df=train_df,report_key_name="missing_columns_within_train_dataset")
            test_df_column_status=self.is_required_columns_exists(base_df=base_df,current_df=test_df,report_key_name="missing_columns_within_test_dataset")
            if train_df_column_status:
                self.data_drift(base_df=base_df,current_df=train_df,report_key_name="data_drift_within_train_dataset")
            if test_df_column_status:
                self.data_drift(base_df=base_df,current_df=test_df,report_key_name="data_drift_within_test_dataset")

            #wite report to yaml file 
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path
                                ,data=self.validation_error)
            data_validation_artifact=artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            return data_validation_artifact

        except Exception as e :
            raise InsuranceException(e,sys)
