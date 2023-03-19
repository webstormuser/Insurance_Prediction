import pymongo
import pandas as pd
import json
from dotenv import load_dotenv
from Insurance_Prediction.config import mongo_client
from Insurance_Prediction.logger import logging

# Provide the mongodb localhost url to connect python to mongodb.


DATABASE_NAME="Insurance"
COLLECTION_NAME="insurance"
DATA_FILE_PATH="/config/workspace/insurance.csv"

if __name__=="__main__":
    df=pd.read_csv(DATA_FILE_PATH)
    print(f" Rows and column in dataframe ,{df.shape}")
    logging.info(f"Rows and Columns in dataframe {df.shape}")
    
    #Convert dataframe into json so that we can dump into mongodb
    df.reset_index(drop=True,inplace=True)
    json_record=list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)