import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import math
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import matplotlib.pylab as plt
from commonLib import *

weather_norm_path = "F:/kdd/dataSets/training/norm_weather (table 7)_training.csv"
weather_info = pd.read_csv(weather_norm_path,encoding='utf-8')

residual_path = "F:/kdd/dataSets/training/residual_norm.csv"
trend_path = "F:/kdd/dataSets/training/trend_norm.csv"
seasonal_path = "F:/kdd/dataSets/training/season_norm.csv"

days = {7:31,8:31,9:30,10:31,11:30,12:31}

data_path = "F:/kdd/dataSets/training/training_20min_avg_volume_update.csv"
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
data = pd.read_csv(data_path, parse_dates=['time_window'], index_col='time_window',date_parser=dateparse)

def writevolume(residual, path):
    fw = open(path, 'w')
    fw.writelines(','.join(['"time_window"', '"residual"']) + '\n')
    l = len(residual)
    indexs = residual.index
    for i in range(l):
        out_line = ','.join(['"' + str(indexs[i]) + '"', '"' + str(residual.loc[indexs[i]][0]) + '"']) + '\n'
        fw.writelines(out_line)
    fw.close()

def getData(id, direction):
    partial_data = data[['volume']][(data['tollgate_id']==id)&(data['direction'] == direction)]
    return partial_data

def get_weather_info():
    weather_dic = {}
    pressures = np.array(weather_info['pressure'])
    sea_pressures = np.array(weather_info['sea_pressure'])
    wind_directions = np.array(weather_info['wind_direction'])
    wind_speeds = np.array(weather_info['wind_speed'])
    temperatures = np.array(weather_info['temperature'])
    rel_humiditys = np.array(weather_info['rel_humidity'])
    precipitations = np.array(weather_info['precipitation'])
    
    date = np.array(weather_info['date'])
    hour = np.array(weather_info['hour'])
    length = len(date)
    for i in range(length):
        if not date[i] in weather_dic:
            weather_dic[date[i]] = {}
        weather_dic[date[i]][hour[i]] = [pressures[i], sea_pressures[i], wind_directions[i], wind_speeds[i], temperatures[i], rel_humiditys[i], precipitations[i]]
    return weather_dic
    
def writeResidual(residual):
    file_path = "F:/kdd/dataSets/training/residual.csv"
    fw = open(file_path,'w')
    fw.writelines(','.join(['"time_window"','"volume"']) + '\n')
    l = len(residual)
    for i in range(l):
        out_line = ','.join([ '"' + str(residual.index[i]) + '"', '"' + str(residual.loc[residual.index[i]][0])+ '"']) + '\n'
        fw.writelines(out_line)
    fw.close()
     
def decompose(ts):  
    decomposition = seasonal_decompose(ts, model = "multiplicative", freq=72)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    seasonal.dropna(inplace=True)
    trend.dropna(inplace=True)
    residual.dropna(inplace=True)
    writeResidual(residual)
    # plt.plot(residual)
    # plt.plot(trend, color='red')
    # plt.show()
    return residual,trend,seasonal
    
def getInfos(id, direction):
    ts = getData(id, direction)
    residual,trend,seasonal = decompose(ts)
    residual_dic = seriestodic(residual)
    trend_dic = seriestodic(trend)
    # seasonal_dic = seriestodic(seasonal)
    return residual_dic,trend_dic,seasonal

def seriestodic(series):
    l = len(series)
    resdic = {}
    for i in range(l):
        resdic[series.index[i]] = series.loc[series.index[i]][0]
    return resdic
    
    
def getVolume(infos):
    time_window = np.array(list(infos.keys()))
    residual = np.array(list(infos.values()))
    l = len(time_window)
    timearr = []
    normtimearr = []
    dates = []
    holidays = []
    weeks = []
    for i in range(l):
        time = time_window[i]
        trace_time = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
        date = str(trace_time.year)+'/'+str(trace_time.month)+'/'+str(trace_time.day)
        week = getweekday(date)
        num = (trace_time.hour*60+trace_time.minute)/20
        if num==0:
            num = 72
        holiday = 0
        if(isholiday(date)):
            holiday = 1
        holidays.append(holiday)
        timearr.append(num)
        dates.append(date)
        weeks.append(week)
    timearr = np.array(timearr)
    normtimearr = zeroNormalize(timearr)
    weeks = np.array(weeks)
    holidays = np.array(holidays)
    # residual = zeroNormalize(residual)
    return timearr,normtimearr,dates,residual,holidays,weeks

def aggregatedata(resdic,path):
    timearr,normtimearr,dates,residual,holidays,weeks = getVolume(resdic)
    weather_dic = get_weather_info()
    weekarr = getweekarr(dates)
    encodingres = encoder(weekarr) 
    encodingres = zeroNormalize(encodingres)
    # print(encodingres)
    columes = []
    for i in range(len(encodingres[0])):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"lastperiod"','"holiday"', '"norm_time"', '"pressure"', '"sea_pressure"',
    '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"', '"volume"'])
    columes = np.hstack((columes, restColumes))
     
    l = len(timearr)
    fw = open(path, 'w')
    fw.writelines(','.join(columes) + '\n') 
    
    lastresiduals = []
    for i in range(l-1):
        lastresiduals.append(residual[i]);
    lastresiduals = zeroNormalize(lastresiduals)    
    
    for i in range(1,l):
        phase,date = getphase(timearr[i],dates[i])
        if not date in weather_dic:
            continue
        if not phase in weather_dic[date]:
            continue
        coder = encodingres[i]
        weather_info = weather_dic[date][phase]
        info = []
        for j in range(len(coder)):
            info.append('"' + str(coder[j]) + '"')
        info = np.array(info)
        restinfo = np.array(['"' + str(lastresiduals[i-1])+ '"','"' + str(holidays[i])+ '"','"' + str(holidays[i])+ '"', '"' + str(normtimearr[i])+ '"',
        '"'+str(weather_info[0])+ '"', '"'+str(weather_info[1])+ '"', '"'+str(weather_info[2])+ '"', '"'+str(weather_info[3])+ '"', '"'+str(weather_info[4])+ '"',
        '"'+str(weather_info[5])+ '"', '"'+str(weather_info[6])+ '"', '"'+str(residual[i])+ '"'])
        info = np.hstack((info,restinfo))
        out_line = ','.join(info)+'\n'    
        fw.writelines(out_line)  
            
            
        # weatherinfo = weather_dic[date][phase]
        # out_line = ','.join([ '"' + str(normtimearr[i]) + '"', '"' + str(holidays[i]) + '"',
                # '"'+str(weatherinfo[0])+ '"', '"'+str(weatherinfo[1])+ '"', '"'+str(weatherinfo[2])+ '"', '"'+str(weatherinfo[3])+ '"', '"'+str(weatherinfo[4])+ '"',
                # '"'+str(weatherinfo[5])+ '"', '"'+str(weatherinfo[6])+ '"', '"'+str(residual[i])+ '"']) + '\n'
        # fw.writelines(out_line)  
    fw.close()
    
def handle_main(id,direction):
    residual_dic,trend_dic,seasonal = getInfos(id,direction)
    aggregatedata(residual_dic, residual_path)
    # print(len(residual_dic))
    aggregatedata(trend_dic, trend_path)
    # print(len(trend_dic))
    writevolume(seasonal,seasonal_path)
    
# def handle_main_new(id, direction):
    # residual_dic,trend_dic,seasonal = getInfos(id,direction)
    # for k,v in residual_dic.i

if __name__=="__main__":
    handle_main(3,1)
    
        
        
