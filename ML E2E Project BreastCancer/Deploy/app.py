# pylint: disable=unused-import
import pandas as pd
from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])

def predict():
    req = request.get_json()
    input_data = req['data']
    input_data_df = pd.DataFrame.from_dict(input_data)

    model = joblib.load('model.pkl')
    


    prediction = model.predict(input_data_df)
    
    if(prediction[0] > 1.0):
        cancer_type = 'Malignant'
    else:
	    cancer_type = 'Bening'

    return jsonify({'output': {'cancer_type':cancer_type}})

@app.route('/')
def home():
    return "Welcome To Breast Cancer Diagnostic Center"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = '3000')





