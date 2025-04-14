# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 22:50:38 2022

@author: G. Rupak
"""
import numpy as np
from flask import Flask, jsonify, url_for, render_template, request
import pickle
from datetime import datetime

# load the model from disk
loaded_model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__, template_folder="template")

@app.route('/')
def welcome():
    return render_template('IndexTryOne.html')

# Result Checker submit HTML page
@app.route('/predict',methods=['POST'])
def predict():
    from_date = ""
    to_date = ""
    
    str_d1 = '2019-10-08'
    from_date = request.form['fromdate']
    
    d1 = datetime.strptime(str_d1, "%Y-%m-%d")
    d2 = datetime.strptime(from_date, "%Y-%m-%d")
    
    delta_start = d2 - d1
    startindex_no = 2106 + delta_start.days
    
    to_date = request.form['todate']
    d3 = datetime.strptime(to_date, "%Y-%m-%d")

    delta_end = d3 - d1
    endindex_no = 2106 + delta_end.days
      
    output = loaded_model.predict(start = startindex_no, end = endindex_no, dynamic = True)
    my_prediction = output.tolist()
    
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run()