import pandas as pd
from datetime import datetime,timedelta
import numpy as np

from getPath import *
pardir = getparentdir()

id = ""
direction = ""

trend_model_path =  pardir+"/dataSets/training/trend.pkl"
seasonal_path =  pardir+"/dataSets/training/season_norm.csv"
volume_test_path =  pardir+"/dataSets/testing_phase1/predict_voulume_data.csv"

def setid_direction(id_, direction_):
    global id
    global direction
    id = id_
    direction = direction_
    
def getseasonal():
    seasonal = pd.read_csv(seasonal_path, encoding='utf-8')
    seasonal_volume = seasonal['residual'][-72:]
    seasonal_volume = list(seasonal_volume)
    res_dic = {}
    res_dic[72] = seasonal_volume[0]
    for i in range(1,72):
        res_dic[i] = seasonal_volume[i]
    return res_dic
    
   
def model_predict(model_path, isResidual,cols=[]):
    predict_data = pd.read_csv(volume_test_path, encoding='utf-8') 
    if len(cols)==0:
        cols = list(predict_data.columns.values)
        
    x = predict_data[cols]
    clf = joblib.load(model_path)
    predict_y = clf.predict(x)
    return predict_y 
    

    