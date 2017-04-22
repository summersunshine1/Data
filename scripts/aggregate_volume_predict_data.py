# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from datetime import datetime
from commonLib import *

days = {7:31,8:31,9:30,10:31,11:30,12:31}
intervals = np.array([18,19,20,21,22,23,45,46,47,48,49,50])
intervals = np.array([24,25,26,27,28,29,51,52,53,54,55,56])


# weather_norm_path = "F:/kdd/dataSets/training/norm_weather (table 7)_training.csv"
# routes_norm_path= "F:/kdd/dataSets/training/norm_widthandlength.csv"
# sources_norm_path = "F:/kdd/dataSets/training/norm_avg_travel_time.csv"

# aggregate_path = "F:/kdd/dataSets/training/totaldata.csv"

weather_norm_path = "F:/kdd/dataSets/testing_phase1/norm_weather (table 7)_test1.csv"

aggregate_path = "F:/kdd/dataSets/testing_phase1/predict_voulume_data.csv"

weather_info = pd.read_csv(weather_norm_path,encoding='utf-8')


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
    
    
def getnormtime():
    dic = {}
    norm_intervals = (intervals - np.mean(intervals)) / np.std(intervals)
    l = len(intervals)
    for i in range(l):
        dic[intervals[i]] = norm_intervals[i]
    return dic
    
def aggregate(id, direction):
    norm_time_dic = getnormtime()
    
    weather_dic = get_weather_info()
    
    dates = np.array(["2016/10/18","2016/10/19","2016/10/20","2016/10/21","2016/10/22","2016/10/23","2016/10/24"]) 
    ids = np.array([id])
    directions = np.array([direction])
    
    datearr = []
    for i in ids:
        for d in directions:
            for date in dates:
                for interval in intervals:
                    datearr.append(date)
    weekarr = getweekarr(datearr)
    encodingres = encoder(weekarr) 
    encodingres = zeroNormalize(encodingres)
    columes = []
    for i in range(len(encodingres[0])):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"id"', '"direction"','"date"','"interval"','"holiday"', '"norm_time"', '"pressure"', '"sea_pressure"',
    '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"'])
    columes = np.hstack((columes, restColumes))
    fw = open(aggregate_path, 'w')
    fw.writelines(','.join(columes) + '\n')
    k = 0
    for i in ids:
        for d in directions:
            for date in dates:
                for interval in intervals:
                    phase,date = getphase(interval,date)
                    if not date in weather_dic:
                        continue
                    if not phase in weather_dic[date]:
                        continue
                        
                    coder = encodingres[k]
                    weather_info = weather_dic[date][phase]
                    info = []
                    for j in range(len(coder)):
                        info.append('"' + str(coder[j]) + '"')
                    info = np.array(info)
                    restinfo = np.array(['"' + str(i) + '"', '"' + str(d) + '"','"' + date + '"', '"' + str(interval) + '"','"' + str(0)+ '"', '"' + str(norm_time_dic[interval])+ '"',
                    '"'+str(weather_info[0])+ '"', '"'+str(weather_info[1])+ '"', '"'+str(weather_info[2])+ '"', '"'+str(weather_info[3])+ '"', '"'+str(weather_info[4])+ '"',
                    '"'+str(weather_info[5])+ '"', '"'+str(weather_info[6])+ '"'])
                    info = np.hstack((info,restinfo))
                    out_line = ','.join(info)+'\n'    
                    fw.writelines(out_line)
                    k+=1
                    # weather_info = weather_dic[date][phase]
                    # out_line = ','.join(['"' + str(i) + '"', '"' + str(d) + '"','"' + date + '"', '"' + str(interval) + '"', '"' + str(norm_time_dic[interval]) + '"','"' + str(0) + '"',
                    # '"' + str(getweekday(date)) + '"','"'+str(weather_info[0])+ '"', '"'+str(weather_info[1])+ '"', '"'+str(weather_info[2])+ '"', '"'+str(weather_info[3])+ '"', '"'+str(weather_info[4])+ '"',
                    # '"'+str(weather_info[5])+ '"', '"'+str(weather_info[6])+ '"']) + '\n'
                    # fw.writelines(out_line)  
    fw.close()
    
def aggregate_main(id, direction):
    aggregate(id,direction)

if  __name__ == "__main__":
    aggregate_main(3,1)
    