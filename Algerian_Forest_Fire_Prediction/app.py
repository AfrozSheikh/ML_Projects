import pickle
import pandas as pd 
import numpy as np
from flask import Flask, render_template , jsonify , request
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

ridge_model = pickle.load(open('./models/ridge.pkl','rb'))
standard_scalar = pickle.load(open('./models/scalar.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_scale_data = standard_scalar.transform([ [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_scale_data)  
        return render_template('home.html' , result=result[0])
    
    else :
        return render_template('home.html')
        


if __name__=="__main__":
    app.run(host='0.0.0.0')
 