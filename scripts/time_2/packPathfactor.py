# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from datetime import datetime
import sys
sys.path.append("F:/kdd/scripts/common")
sys.path.append('E:/kdd/Data/scripts/common')

from commonLib import *
from handlePath import *

weather_norm_path = ""
sources_norm_path = ""
aggregate_path = ""

weather_info = ""
sources_info = ""

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
    links = getlinks()
    columes = []
    columes.append('"linkid"')
    for i in range(7):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"dateandtime"','"holiday"', '"norm_time"', '"pressure"', '"sea_pressure"',
    '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"', '"avg_travel_time"'])
    columes = np.hstack((columes, restColumes))
    fw = open(aggregate_path, 'w')
    fw.writelines(','.join(columes) + '\n')
    for link in links:
        # ahead_links = linkout[link]
        # a_times = []
        # for a_link in ahead_links:
            # a_times.append(sources_info['avg_travel_time'][sources_info['linkid']==int(a_link)])
        # a_times = np.mean(a_times, axis = 0) 

        # a_times = zeroNormalize(a_times)
        dates = np.array(sources_info['date'][sources_info['linkid']==link])
        avg_travel_times = np.array(sources_info['avg_travel_time'][sources_info['linkid']==link]) 
        # avg_travel_times = zeroNormalize(avg_travel_times)
        norm_times = np.array(sources_info['norm_time'][sources_info['linkid']==link])
        time_intervals = np.array(sources_info['interval'][sources_info['linkid']==link]) 
        length = len(dates)
        holidaydic = getNormalizeHoliday(dates)
        weekarr = getweekarr(dates)
        encodingres = encoder(weekarr)
        encodingres = zeroNormalize(encodingres)
        
        for i in range(length):
            phase,date = getphase(time_intervals[i], dates[i])
            if not date in weather_dic:
                continue
            if not phase in weather_dic[date]:
                continue
            coder = encodingres[i]
            weather_info = weather_dic[date][phase]
            info = [link]
            for j in range(len(coder)):
                info.append('"' + str(coder[j]) + '"')
            info = np.array(info)
            dateandtime = dates[i]+"-"+str(time_intervals[i])
            restinfo = np.array(['"' + dateandtime+ '"', '"' + str(holidaydic[date])+ '"', '"' + str(norm_times[i])+ '"',
            '"'+str(weather_info[0])+ '"', '"'+str(weather_info[1])+ '"', '"'+str(weather_info[2])+ '"', '"'+str(weather_info[3])+ '"', '"'+str(weather_info[4])+ '"',
            '"'+str(weather_info[5])+ '"', '"'+str(weather_info[6])+ '"','"'+str(avg_travel_times[i])+ '"'])
            info = np.hstack((info,restinfo))
            out_line = ','.join(info)+'\n'
            
            # out_line = ','.join(['"' + str(norm_times[i]) + '"', '"' + str(isholiday(date)) + '"', '"' + str(getweekday(date)) + '"'
            # '"' + str(weekarr[0]) + '"', '"' + str(weekarr[1]) + '"','"' + str(weekarr[2]) + '"','"' + str(weekarr[3]) + '"','"' + str(weekarr[4]) + '"',
            # '"' + str(weekarr[5]) + '"','"' + str(weekarr[6]) + '"',
            # '"' + str(routes_dic[id]['average_width'])+ '"','"'+str(routes_dic[id]['total_length'])+ '"',
            # '"'+str(weather_info[0])+ '"', '"'+str(weather_info[1])+ '"', '"'+str(weather_info[2])+ '"', '"'+str(weather_info[3])+ '"', '"'+str(weather_info[4])+ '"',
            # '"'+str(weather_info[5])+ '"', '"'+str(weather_info[6])+ '"', '"'+str(avg_travel_times[i])+ '"']) + '\n'
            fw.writelines(out_line)  
    fw.close()
    
def aggregate_main(isVal):
    global weather_norm_path
    global sources_norm_path
    global aggregate_path

    global weather_info
    global sources_info
    if not isVal:
        weather_norm_path = "F:/kdd/dataSets/training/norm_weather (table 7)_training.csv"
        sources_norm_path = "F:/kdd/dataSets/training/training_20min_avg_path_travel_time.csv"
        aggregate_path = "F:/kdd/dataSets/training/traveltime_totaldata.csv"
    else:
        weather_norm_path = "F:/kdd/dataSets/testing_phase1/norm_weather (table 7)_test1.csv"
        sources_norm_path = "F:/kdd/dataSets/testing_phase1/test1_20min_avg_path_travel_time.csv"
        aggregate_path = "F:/kdd/dataSets/testing_phase1/traveltime_totaldata.csv"
    
    weather_info = pd.read_csv(weather_norm_path,encoding='utf-8')
    sources_info = pd.read_csv(sources_norm_path,encoding='utf-8')
    aggregate()
        
if  __name__ == "__main__":
    aggregate_main(1)