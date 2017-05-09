# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from datetime import datetime

import sys
sys.path.append("F:/kdd/scripts/common")
from commonLib import *
from handlePath import *

weather_norm_path = "F:/kdd/dataSets/testing_phase1/norm_weather (table 7)_test1.csv"
aggregate_path = "F:/kdd/dataSets/testing_phase1/predict_path_data.csv"
weather_info = pd.read_csv(weather_norm_path,encoding='utf-8')

intervals = []

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
    
def aggregate():
    weather_dic = get_weather_info()
    norm_time_dic = getnormtime(intervals)
    columes = []
    for i in range(7):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"date"','"interval"','"holiday"', '"norm_time"', '"pressure"', '"sea_pressure"',
    '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"'])
    columes = np.hstack((columes, restColumes))
    fw = open(aggregate_path, 'w')
    fw.writelines(','.join(columes) + '\n')
    
    dates = np.array(["2016/10/18","2016/10/19","2016/10/20","2016/10/21","2016/10/22","2016/10/23","2016/10/24"])
    datearr = []
    intervalarr = []
    for date in dates:
        for interval in intervals:
            datearr.append(date)
            intervalarr.append(interval)
    holidaydic = getNormalizeHoliday(datearr)
    weekarr = getweekarr(datearr)
    encodingres = encoder(weekarr)
    encodingres = zeroNormalize(encodingres) 
    length = len(datearr)
    
    for i in range(length):
        interval = intervalarr[i]
        date = datearr[i]
        phase,date = getphase(intervalarr[i],datearr[i])
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
        restinfo = np.array(['"' + date + '"','"' + str(interval) + '"','"' + str(holidaydic[date]) + '"','"' + str(norm_time_dic[interval]) + '"',
        '"'+str(weather_info[0])+ '"', '"'+str(weather_info[1])+ '"', '"'+str(weather_info[2])+ '"', '"'+str(weather_info[3])+ '"', '"'+str(weather_info[4])+ '"',
        '"'+str(weather_info[5])+ '"', '"'+str(weather_info[6])+ '"'])
        info = np.hstack((info,restinfo))
        out_line = ','.join(info)+'\n'
        
        fw.writelines(out_line)  
    fw.close()
    
    
def aggregate_predict_path_main(isVal):
    global intervals
    if isVal:
        intervals = np.array([18,19,20,21,22,23,45,46,47,48,49,50])
    else:
        intervals = np.array([24,25,26,27,28,29,51,52,53,54,55,56])
    aggregate()
        
if  __name__ == "__main__":
    aggregate_predict_path_main(0)