# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from datetime import datetime,timedelta

days = {7:31,8:31,9:30,10:31,11:30,12:31}

data_path = ""

result_path = "F:/kdd/dataSets/testing_phase1/predict_result.csv"

def get_source_data(isValidation):
    data = pd.read_csv(data_path,encoding='utf-8')
    # X = data[['norm_time','holiday', 'week','average_width','total_length','pressure','sea_pressure','wind_direction','temperature','rel_humidity','precipitation']]
    tempcols = []
    # for i in range(20):
        # tempcols.append(str(i))
    tempcols.append(str(1))
    tempcols = np.array(tempcols)
    # restcols = np.array(['average_width','total_length','pressure','sea_pressure','wind_direction','temperature','rel_humidity'])
    restcols = np.array(['norm_time', 'average_width','total_length','pressure','sea_pressure','wind_direction','temperature','rel_humidity'])
    cols = np.hstack((tempcols, restcols))
    X = data[cols]
    
    if isValidation:
        y = data['avg_travel_time'] 
        return X,y
    else:
        return X
    
def my_custom_loss_func(ground_truth, predictions):
    return np.mean(np.abs(ground_truth-predictions)/ground_truth)
    
def predict(isValidation):
    rf = joblib.load('F:/kdd/scripts/rf.pkl')
    svr = joblib.load('F:/kdd/scripts/svr.pkl')
    gbt = joblib.load('F:/kdd/scripts/gbt.pkl')
    if isValidation:
        X,y = get_source_data(isValidation)
        predict_y1 = rf.predict(X)
        predict_y2 = svr.predict(X)
        predict_y3 = gbt.predict(X)
        predict_y = (predict_y1+predict_y2+predict_y3)/3
        print(my_custom_loss_func(y,predict_y))
    else:
        X = get_source_data(isValidation)
        predict_y1 = rf.predict(X)
        predict_y2 = svr.predict(X)
        predict_y3 = gbt.predict(X)
        predict_y = (predict_y1+predict_y2+predict_y3)/3
        return predict_y
      
def get_time_from_interval(date,interval):
    hour = int(interval*20/60)
    minute = int(interval*20%60)
    trace_time = datetime.strptime(date, "%Y/%m/%d")
    year = trace_time.year
    month = trace_time.month
    day = trace_time.day
    # if hour == 24:
        # hour = 0
        # day+=1
        # if day>days[month]:
            # month+=1
    start_time_window = datetime(year, month, day, hour, minute, 0)
    end_time_window = start_time_window + timedelta(minutes=20)
    return start_time_window,end_time_window

def format_result(y):
    fw = open(result_path, 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
    data = pd.read_csv(data_path,encoding='utf-8')
    x = data[['id', 'date', 'interval','average_width','total_length','pressure','sea_pressure','wind_direction','temperature','rel_humidity','precipitation']]
    l = len(y)
    print(l)
    for i in range(l):
        id = x['id'][i]
        idarr = id.split('-')
        start_time_window,end_time_window = get_time_from_interval(x['date'][i], x['interval'][i])
        out_line = ','.join(['"' + idarr[0] + '"', '"' + idarr[1] + '"',
                                 '"[' + str(start_time_window) + ',' + str(end_time_window) + ')"',
                                 '"' + str(y[i]) + '"']) + '\n'
        fw.writelines(out_line)
    fw.close()    
 
def predict_main(isValidation):
    global data_path
    if isValidation:
        data_path = "F:/kdd/dataSets/testing_phase1/totaldata.csv"
        predict(isValidation)
    else:
        data_path = "F:/kdd/dataSets/testing_phase1/predict_data.csv"
        y = predict(isValidation)
        format_result(y)
    

if __name__ == "__main__":
    predict_main(0)

    