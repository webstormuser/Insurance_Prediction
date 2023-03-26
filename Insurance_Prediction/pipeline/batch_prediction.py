from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.logger import logging
from Insurance_Prediction.predictor import ModelResolver
import pandas as pd 
from Insurance_Prediction.utils import load_object
import os,sys
from datetime import datetime
PREDICTION_DIR="prediction"
import numpy as np
from Insurance_Prediction.pipeline.training_pipeline import start_training_pipeline

def start_batch_prediction(input_file_path):
    '''
        This function is used for creating prediction for complete dataset/file provided by user and creating prediction file
    '''
            try:
                #start_training_pipeline()
                os.makedirs(PREDICTION_DIR,exist_ok=True)
                logging.info(f"Creating model resolver object")
                model_resolver = ModelResolver(model_registry="saved_models")
                logging.info(f"Reading file :{input_file_path}")
                df = pd.read_csv(input_file_path)

                logging.info(f"Loading transformer to transform dataset")
                transformer = load_object(file_path=model_resolver.get_latest_transformer_path())

                input_feature_names =  list(transformer.feature_names_in_)
                input_arr = transformer.transform(df[input_feature_names])

                logging.info(f"Loading model to make prediction")
                model = load_object(file_path=model_resolver.get_latest_model_path())
                prediction = model.predict(input_arr)

                df["predicted_insurace"]=prediction
                prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
                prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
                df.to_csv(prediction_file_path,index=False,header=True)
                return prediction_file_path
            except Exception as e:
                raise InsuranceException(e, sys)


class PredictPipeline:
    '''
     This class predict the Insurance amount for a person as per input provided by customer
    '''
    def __init__(self):pass

    def predict(self,features):
        try:
            model_resolver = ModelResolver(model_registry="saved_models")
            transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
            model = load_object(file_path=model_resolver.get_latest_model_path())
            data_scaled=transformer.transform(features)
            prediction = model.predict(data_scaled)
            return prediction

            
        except Exception as e :
            raise InsuranceException(e,sys)

class CustomData:
    def __init__(self, age:int,sex:str,bmi:float,children:int,smoker:str,region:str,
    expenses:float):
        self.age=age
        self.sex=sex
        self.bmi=bmi
        self.children=children
        self.smoker=smoker
        self.region=region
        self.expenses=expenses

    def get_data_as_data_frame(self):
        try:
            custom_data_as_input_dict={
                "age":[self.age],
                "sex":[self.sex],
                "bmi":[self.bmi],
                "children":[self.children],
                "smoker":[self.smoker],
                "region":[self.region],
                "expenses":[self.expenses]
            }
            return pd.DataFrame(custom_data_as_input_dict)

        except Exception as e:
                raise InsuranceException(e,sys)