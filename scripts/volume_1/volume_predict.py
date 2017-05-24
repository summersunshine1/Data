import pandas as pd
from datetime import datetime,timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.externals import joblib

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import arma_order_select_ic
import os.path
import tensorflow as tf

from timeseriescase import *
from predictResidual import *

residual_model_path = "F:/kdd/dataSets/training/residual.pkl"
trend_model_path = "F:/kdd/dataSets/training/trend.pkl"
seasonal_path = "F:/kdd/dataSets/training/season_norm.csv"

volume_test_path = "F:/kdd/dataSets/testing_phase1/predict_voulume_data.csv"
real_volume_path = "F:/kdd/dataSets/testing_phase1/test1_20min_avg_volume_update.csv"
final_res_path = "F:/kdd/dataSets/testing_phase1/predicted_volume.csv"
error_residual_path = "F:/kdd/dataSets/training/error_residual.csv"

lstm_predict_path = "F:/kdd/dataSets/training/lstmResult.csv"
residual_data = pd.read_csv(lstm_predict_path, encoding='utf-8')


predict_data = pd.read_csv(volume_test_path, encoding='utf-8') 
real_data = pd.read_csv(real_volume_path, encoding='utf-8')

pastresidual = pd.read_csv("F:/kdd/dataSets/training/residual.csv", encoding='utf-8')

id = 3
direction = 1

def getresidual():
    volumes = pastresidual['volume']
    # plt.plot(volumes)
    # plt.show()
    # temp1 = volumes[-35:]
    # temp2 = volumes[-72:-35]
    # print(len(temp1))
    # print(len(temp2))
    # temp = np.hstack((temp1, temp2))
    temp = volumes[-72:]
    temp = np.array(temp)
    # for i in range(72):
        # print(temp[i])
    return temp

def setid_direction(id_, direction_):
    global id
    global direction
    id = id_
    direction = direction_

def gettruevolume():
    time_windows = real_data['time_window'][(real_data['tollgate_id']==id)&(real_data['direction'] == direction)]
    # print(time_windows)
    volumes = real_data['volume'][(real_data['tollgate_id']==id)&(real_data['direction'] == direction)]
    l = len(time_windows)
    time_windows = np.array(time_windows)
    volumes = np.array(volumes)
    
    res_dic = {}
    for i in range(l):
        time = time_windows[i]
        trace_time = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
        date = str(trace_time.year)+'/'+str(trace_time.month)+'/'+str(trace_time.day)
        num = (trace_time.hour*60+trace_time.minute)/20
        if num==0:
            num = 72
        if not date in res_dic:
            res_dic[date]={}
        res_dic[date][num] = volumes[i]
    return res_dic,time_windows    
    
def model_predict(model_path, isResidual,cols):
    # if isResidual:
        # x = predict_data[['norm_time','holiday','week','pressure', 'sea_pressure',
    # 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation']]
    # else:
        # tempcols = []
        # tempcols.append(str(5))
        # tempcols = np.array(tempcols)
        # restcols = np.array(['average_width','total_length','pressure','sea_pressure','wind_direction','temperature','rel_humidity'])
        # restcols = np.array(['holiday','norm_time','temperature'])
        # cols = np.hstack((tempcols, restcols))
        # x = predict_data[['norm_time','holiday','week','pressure', 'sea_pressure',
    # 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation']]
    x = predict_data[cols]
    clf = joblib.load(model_path)
    predict_y = clf.predict(x)
    return predict_y 
    
def getseasonal(seasonal):
    seasonal_volume = seasonal['residual'][-72:]
    seasonal_volume = list(seasonal_volume)
    res_dic = {}
    res_dic[72] = seasonal_volume[0]
    for i in range(1,72):
        res_dic[i] = seasonal_volume[i]
    return res_dic
    
def getnonseasonal(seasonal_dic,true_volume_dic, time_windows):
    l = len(time_windows)
    nonseasonal = []
    part = []
    for i in range(l):
        time = time_windows[i]
        trace_time = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
        date = str(trace_time.year)+'/'+str(trace_time.month)+'/'+str(trace_time.day)
        num = (trace_time.hour*60+trace_time.minute)/20
        if num==0:
            num = 72
        
        if i%6==0 and not i == 0:
            nonseasonal.append(part)
            part = []
        part.append(true_volume_dic[date][num] - seasonal_dic[num])
    nonseasonal.append(part)
    nonseasonal = np.array(nonseasonal)
    return nonseasonal
    
def average(series, n=None):
    if n is None:
        return average(series, len(series))
    return float(sum(series[-n:]))/n
    
def get_nonseasonal_avg(seasonal_dic,true_volume_dic, time_windows):
    nonseasonals = getnonseasonal(seasonal_dic,true_volume_dic, time_windows)
    # print(nonseasonals)
    res = []
    for i in range(len(nonseasonals)):
        nonseasonal = nonseasonals[i]
        prevseq = nonseasonal
        for j in range(len(nonseasonal)):
            v = average(prevseq)
            res.append(v)
            prevseq = np.hstack((prevseq[1:],v))
    return res    
    
def getResidualdic():
    dates = residual_data['date']
    intervals = residual_data['interval']
    volumes = residual_data['volume']
    res_dic = {}
    l = len(dates)
    for i in range(l):
        if not dates[i] in res_dic:
            res_dic[dates[i]] = {}
        # if not intervals[i] in res_dic[dates[i]]:
            # res_dic[dates[i]][] = {}
        res_dic[dates[i]][intervals[i]] = volumes[i]
    return res_dic
    
def predict(trend_cols,residual_cols):
    residual_predict_y = model_predict(residual_model_path, 1,residual_cols)
    # residual_dic = getResidualdic()
    # residual_predict_y = getresidual()
    trend_predict_y = model_predict(trend_model_path, 0, trend_cols)
    # plt.plot(residual_predict_y)
    # plt.plot(trend_predict_y, color='red')
    # plt.show()
    
    predict_intervals = predict_data['interval']
    predict_dates = predict_data['date']
    
    seasonal = pd.read_csv(seasonal_path, encoding='utf-8')
    seasonal_dic = getseasonal(seasonal)
    
    true_volume_dic, time_windows= gettruevolume()
    
    l = len(trend_predict_y)
    predicted_ys = []
    ground_ys = []
    d = []
    non_seasonal = get_nonseasonal_avg(seasonal_dic,true_volume_dic, time_windows)
    # print(non_seasonal)
    for i in range(l):
        if not predict_dates[i] in true_volume_dic:
            continue
        if not predict_intervals[i] in true_volume_dic[predict_dates[i]]:
            continue
        temp_interval = predict_intervals[i]
        if predict_intervals[i]==72:
            temp_interval = 0
            
        
        predicted_y1 =  trend_predict_y[i] + seasonal_dic[predict_intervals[i]]#+residual_predict_y[i]#+residual_dic[predict_dates[i]][predict_intervals[i]]
        # if id == 2 and direction == 0:
            # predicted_y1 = seasonal_dic[predict_intervals[i]]*non_seasonal[i]
        # predicted_y =(predicted_y2+predicted_y2)/2
        predicted_ys.append(predicted_y1)
        ground_ys.append(true_volume_dic[predict_dates[i]][predict_intervals[i]])
        d.append(time_windows[i])
    predicted_ys = np.array(predicted_ys)
    ground_ys = np.array(ground_ys)
    d = np.array(d)
    return predicted_ys,ground_ys,d
    
def my_custom_loss_func(ground_truth, predictions):
    return np.mean(np.abs(ground_truth-predictions)/ground_truth)
    
def get_time_from_interval(date,interval):
    hour = int(interval*20/60)
    minute = int(interval*20%60)
    trace_time = datetime.strptime(date, "%Y/%m/%d")
    year = trace_time.year
    month = trace_time.month
    day = trace_time.day
    if hour == 24:
        hour = 0
        # day+=1
        # if day>days[month]:
            # month+=1
    start_time_window = datetime(year, month, day, hour, minute, 0)
    end_time_window = start_time_window + timedelta(minutes=20)
    return start_time_window,end_time_window
    
def writeResTofile(trend_cols,residual_cols):
    predicted_ys, predict_intervals, predict_dates = newPredict(trend_cols,residual_cols)
    if not os.path.exists(final_res_path):
        fw = open(final_res_path,'w')
        fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"'])+'\n')
    else:
        fw = open(final_res_path,'a')
    l = len(predicted_ys)
    for i in range(l):
        start_time_window,end_time_window = get_time_from_interval(predict_dates[i], predict_intervals[i])
        out_line = ','.join(['"' + str(id) + '"', '"[' + str(start_time_window) + ',' + str(end_time_window) + ')"', 
        '"' + str(direction) + '"', '"' + str(int(predicted_ys[i])) + '"',]) + '\n'
        fw.writelines(out_line)
    fw.close()
           
    
def newPredict(trend_cols,residual_cols):
    # residual_predict_y = model_predict(residual_model_path, 0)
    # residual_dic = getResidualdic()#直接根据先前记录将时间序列都预测出来
    # residual_dic = predictfromData(id, direction)#根据测试前2个小时记录预测后两个小时
    # trend_predict_y = model_predict(trend_model_path, 0)
    residual_predict_y = model_predict(residual_model_path, 1,residual_cols)
    trend_predict_y = model_predict(trend_model_path, 0, trend_cols)
    predict_intervals = predict_data['interval']
    predict_dates = predict_data['date'] 
    seasonal = pd.read_csv(seasonal_path, encoding='utf-8')
    seasonal_dic = getseasonal(seasonal)
    l = len(trend_predict_y)
    true_volume_dic, time_windows= gettruevolume()
    non_seasonal = get_nonseasonal_avg(seasonal_dic,true_volume_dic, time_windows)
    print(len(non_seasonal))
    predicted_ys = []
    residual_error = gettrain_error(id, direction)
    for i in range(l):
        predicted_y1 =  trend_predict_y[i]+seasonal_dic[predict_intervals[i]]+residual_error[i]#*residual_predict_y[i]#+residual_dic[predict_dates[i]][predict_intervals[i]]
        # predicted_y2 =  trend_predict_y[i] + seasonal_dic[predict_intervals[i]]+residual_predict_y[i]
        # predicted_y3 =  seasonal_dic[predict_intervals[i]]+non_seasonal[i]
        # predicted_y =(predicted_y1+predicted_y3)/2
        # if id == 2 and direction == 0:
            # predicted_y1 = seasonal_dic[predict_intervals[i]]*non_seasonal[i]
        
        predicted_ys.append(predicted_y1)
    predicted_ys = np.array(predicted_ys)
    return predicted_ys, predict_intervals, predict_dates

def writeResidualToFile(filepath, rate, dates, id, direction):
    if not os.path.exists(filepath):
        fw = open(filepath,'w')
        fw.writelines(','.join(['"id"', '"direction"', '"time_window"', '"residual"'])+'\n')
    else:
        fw = open(filepath,'a')
    for i in range(len(rate)):
        out_line = ','.join(['"' + str(id) + '"','"' + str(direction) + '"','"' + dates[i] + '"','"' + str(rate[i]) + '"']) + '\n'
        fw.writelines(out_line)
    fw.close() 
    
def predict_main(id,direction,trend_cols,residual_cols, isvalid):
    setid_direction(id, direction)
    if not isvalid:
        writeResTofile(trend_cols,residual_cols)
    else:
        predicted_ys,ground_ys,d = predict(trend_cols,residual_cols)
        rate = ground_ys-predicted_ys
        writeResidualToFile(error_residual_path, rate, d, id, direction)
        
        v1 = pd.DataFrame(predicted_ys)
        v2 = pd.DataFrame(ground_ys) 
        t = pd.DatetimeIndex(d)
        v1.index = t
        v2.index = t
        plt.plot(v1)
        plt.plot(v2,color = 'red')
        plt.show()
        print(my_custom_loss_func(ground_ys, predicted_ys))
    
    
    
if __name__ == "__main__":
    predict_main(3,1)
    
    
        
    
    
 
    
    
      