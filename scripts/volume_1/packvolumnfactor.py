# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from datetime import datetime,timedelta
import math
from commonLib import *

# old_volumn = "F:/kdd/dataSets/training/volume(table 6)_training.csv"
# old_volumn = "F:/kdd/dataSets/testing_phase1/volume(table 6)_test1.csv"

volumn_path = "F:/kdd/dataSets/training/training_20min_avg_volume.csv"
weather_norm_path = "F:/kdd/dataSets/training/norm_weather (table 7)_training.csv"

volumn_infos = pd.read_csv(volumn_path,encoding='utf-8')
weather_info = pd.read_csv(weather_norm_path,encoding='utf-8')

data_path = "F:/kdd/dataSets/training/voulumn_aggregate_data.csv"

tollgates_weight={1:2,2:1,3:3}

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
    
def zeroNormalize(arr):
    mu = np.average(arr)
    sigma = np.std(arr)
    return (arr-mu)/sigma

def Normalize_volumn_info(id, direction):
    time_windows = np.array(volumn_infos['time_window'])
    volumes = np.array(volumn_infos['volume'][])
    # volumes = zeroNormalize(volumes)
    tollgate_ids = np.array(volumn_infos['tollgate_id'])
    directions = np.array(volumn_infos['direction'])
    
    length = len(time_windows)
    timearr = []
    dates = []
    tollgate_weights = []
    for i in range(length):
        timepair = time_windows[i].split(',')
        endtime = timepair[1]
        starttime = timepair[0]
        trace_endtime = datetime.strptime(endtime, "%Y-%m-%d %H:%M:%S)")
        trace_starttime = datetime.strptime(starttime, "[%Y-%m-%d %H:%M:%S")
        date = str(trace_starttime.year)+'/'+str(trace_starttime.month)+'/'+str(trace_starttime.day)
            
        num = (trace_starttime.hour*60+trace_starttime.minute)/20
        if num==0:
            num = 72
        timearr.append(num)
        dates.append(date)
        tollgate_weights.append(tollgates_weight[tollgate_ids[i]])   
        
    timearr = np.array(timearr)
    normtimearr = zeroNormalize(timearr)
    directions = zeroNormalize(directions)
    dates = np.array(dates)
    holidaydic = getNormalizeHoliday(dates)
    weekdaydic = getNormalizeWeekday(dates)
    tollgate_weights = zeroNormalize(tollgate_weights)
    return timearr,normtimearr,dates,volumes,tollgate_weights,directions,holidaydic,weekdaydic 
    
def aggregate():
    timearr,normtimearr,dates,volumes,tollgate_weights,directions,holidaydic,weekdaydic  = Normalize_volumn_info()
    weather_dic = get_weather_info()
    fw = open(data_path, 'w')
    # fw.writelines(','.join(['"tollgate_weight"', '"norm_time"','"holiday"', '"week"', '"direction"', '"pressure"', '"sea_pressure"',
    # '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"', '"volume"']) + '\n')
    fw.writelines(','.join(['"norm_time"','"holiday"', '"week"', '"pressure"', '"sea_pressure"',
    '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"', '"volume"']) + '\n')
    l = len(dates)
    for i in range(l):
        phase,date = getphase(timearr[i],dates[i])
        if not date in weather_dic:
            continue
        if not phase in weather_dic[date]:
            continue
        weatherinfo = weather_dic[date][phase]
        out_line = ','.join(['"' + str(tollgate_weights[i]) + '"', '"' + str(normtimearr[i]) + '"', '"' + str(holidaydic[date]) + '"', '"' + str(weekdaydic[date]) + '"', '"' + str(directions[i]) + '"',
                '"'+str(weatherinfo[0])+ '"', '"'+str(weatherinfo[1])+ '"', '"'+str(weatherinfo[2])+ '"', '"'+str(weatherinfo[3])+ '"', '"'+str(weatherinfo[4])+ '"',
                '"'+str(weatherinfo[5])+ '"', '"'+str(weatherinfo[6])+ '"', '"'+str(float(volumes[i]))+ '"']) + '\n'
        fw.writelines(out_line)  
    fw.close()    
    
if __name__=="__main__":
    aggregate()
        
    
    
