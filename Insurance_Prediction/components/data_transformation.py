from Insurance_Prediction import utils
from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.logger import logging
from Insurance_Prediction.entity import artifact_entity 
from Insurance_Prediction.entity import config_entity
import os,sys
import pandas as pd 
import numpy as np
from Insurance_Prediction import utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from Insurance_Prediction.config import TARGET_COLUMN

class DataTransformation:
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise InsuranceException(e,sys)

   

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            cat_features=['sex', 'smoker', 'region']
            num_features=['age', 'bmi', 'children']
            numeric_transformer =Pipeline(steps=[('robust_scaler',RobustScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
            preprocessor = ColumnTransformer(
                transformers=[('num', numeric_transformer, num_features),
                              ('cat', categorical_transformer, cat_features)])
            pipeline=Pipeline([('preprocessor',preprocessor)])
            return pipeline
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            # reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #selecting input feature for train and test dataframe
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            #selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)

            #transforming input features
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)

            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipleine)

            data_transformation_artifact=artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path
            )
            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact


        except Exception as e:
            raise InsuranceException(e,sys)

        

    