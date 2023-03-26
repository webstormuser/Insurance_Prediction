from Insurance_Prediction.logger import logging
from Insurance_Prediction.exception import InsuranceException
import os,sys
from Insurance_Prediction.pipeline.training_pipeline import start_training_pipeline
from Insurance_Prediction.pipeline.batch_prediction import start_batch_prediction,CustomData
from Insurance_Prediction.pipeline.batch_prediction import CustomData,PredictPipeline
file_path="/config/workspace/insurance.csv"

if __name__=="__main__":
    try:
       
        start_training_pipeline()
        output_file=start_batch_prediction(input_file_path=file_path)
        print(output_file)
    except Exception as e:
        print(e)