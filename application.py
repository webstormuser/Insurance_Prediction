from flask import Flask ,request,render_template
from flask_cors import CORS,cross_origin
from Insurance_Prediction.exception import InsuranceException
import os,sys
import os
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
#from Insurance_Prediction.pipeline.training_pipeline import start_training_pipeline
from Insurance_Prediction.pipeline.batch_prediction import CustomData,start_batch_prediction,PredictPipeline
from dotenv import load_dotenv
load_dotenv()

application= Flask(__name__)
app=application

@app.route('/', methods=['GET'])  # route to display the Home page
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/prediction',methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            age=int(request.form.get('age')),
            sex=request.form.get('sex'),
            bmi=float(request.form.get('bmi')),
            children=int(request.form.get('children')),
            smoker=request.form.get('smoker'),
            region=request.form.get('region'),
            expenses=float(request.form.get('expenses'))
            
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)
        return render_template('home.html',result=result[0])

if __name__ == '__main__': 
    try:
        #app.run(host='0.0.0.0',port=5002)
        app.run(host='0.0.0.0')
    except Exception as e:
        raise InsuranceException(e,sys)