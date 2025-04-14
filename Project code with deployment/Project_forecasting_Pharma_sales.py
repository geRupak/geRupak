# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 17:04:55 2022

@author: G. Rupak
"""

#importing required packages
import mysql.connector
import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as stats
import dtale
from sklearn.pipeline import Pipeline
from autots import AutoTS

import pickle
import dill as pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

############################################################3

#establishing the connection
db = mysql.connector.connect(host="Localhost",user="root",password="8888",database="data")
# if their is a error "authentication pulgin error uninstall mysql-connector and install mysql-connector-python or 2 solu cnx = mysql.connector.connect(user='root', password='',host='127.0.0.1', database='airpaw',auth_plugin='mysql_native_password')

### Fetching whole data

data = pd.read_sql_query(" select * from dailysa ",db)

"""
Auto Exploratory Data Analysis
                                """
############################################################## 

def EDA():
    d = dtale.show(data)
    d.open_browser()
   
"""
    Data  Preprocessing
                         """
#########################################
                                                  
In this dataset, We dont have " Missing Values " (According to the report from create_report())

def Preprocessing():
    
    data['datum'] = pd.to_datetime(data['datum'], format= '%m/%d/%Y')
   
    data.drop_duplicates(inplace=True)
    
    data['M01AB'] = data['M01AB'].astype(float)
    data['M01AE'] = data['M01AE'].astype(float)
    data['N02BA'] = data['N02BA'].astype(float)
    data['N02BE'] = data['N02BE'].astype(float)
    data['N05B'] = data['N05B'].astype(float)
    data['N05C'] = data['N05C'].astype(float)
    data['R03'] = data['R03'].astype(float)
    data['R06'] = data['R06'].astype(float)
    
    cols = data.columns[1:]
    
    for col in cols:
        
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        
        iqr = q3 - q1
        
        upper_whisker = q3 + (1.5*iqr)
        lower_whisker = q1 - (1.5*iqr)
        
        data[col] = np.where(data[col]>upper_whisker, upper_whisker,
                 np.where(data[col]<lower_whisker, lower_whisker, data[col]))

##############################
"    Model   "
 
def model_build(drug_name):
    # Data Partition in Train & Test
    train = data[-365:]
    model = AutoTS(forecast_length=7,frequency='infer',prediction_interval=0.95,ensemble=None,model_list="probabilistic",transformer_list="fast",drop_most_recent=1,max_generations=4,num_validations=2,validation_method="backwards")
    model = model.fit(train,date_col="datum" ,value_col=drug_name,id_col=None )
    
    # Predicting
    prediction = model.predict()
    forecast = pd.DataFrame(prediction.forecast)
    
    # plot a sample
    prediction.plot(model.df_wide_numeric,
                    series=model.df_wide_numeric.columns[0],
                    start_date=data["datum"].iloc[-30])
    
    # accuracy of all tried model results
    model_results = model.results()
    model_results = pd.DataFrame(model_results)
  
    # and aggregated from cross validation
    validation_results = model.results("validation")
    
    pickle.dump(model, open('model3.pkl', 'wb'))  
    model = pickle.load(open('model3.pkl', 'rb'))

    return model,model_results,validation_results,forecast


######################################

" Pipeline "
pipe = Pipeline([("EDA",EDA()),("Preprocessing",Preprocessing()),("Model",model_build("M01AB"))])


########################################



future_predictions_best = model.predict(testdata = test.M01AB, model = 'best')

import auto_ts

from auto_ts import auto_timeseries

def model_build(drug_name):
    train = data.head(2099)
    test = data.tail(7)

    m = auto_timeseries(score_type='rmse', time_interval='D', non_seasonal_pdq = None, seasonality = True,seasonal_period=7, model_type=['best'], verbose = 2)
    m = m.fit(traindata = train, ts_column = "datum", target = drug_name, cv = 3)

    m_rmse = m.get_leaderboard()

    results_dict = m.get_ml_dict()

    future_predictions_prophet = m.predict(testdata = test, model = 'Prophet') 

    test["Prophet"] = future_predictions_prophet["yhat"].values

    pickle.dump(m, open('model.pkl', 'wb'))  
    m = pickle.load(open('model.pkl', 'rb'))
    
    return m,m_rmse,future_predictions_prophet

" Pipeline "
pipe = Pipeline([("EDA",EDA()),("Preprocessing",Preprocessing()),("Model",model_build("M01AB"))])










