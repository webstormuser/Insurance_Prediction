from flask import Flask ,request,render_template
from flask_cors import CORS,cross_origin
import os
import pandas as pd 
import numpy as np
import sklearn

app = Flask(__name__)

@app.route('/', methods=['GET'])  # route to display the Home page
@cross_origin()
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True,port=5001,host='0.0.0.0')