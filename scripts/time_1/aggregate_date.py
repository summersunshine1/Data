# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from datetime import datetime
from getPath import *
pardir = getparentdir()
commonpath = pardir+'/scripts/common'
import sys
sys.path.append(commonpath)
from commonLib import *
from convertcontinuous import *

globalcolumes = ["pressure", "wind_direction", "wind_speed", "temperature", "rel_humidity"]

weather_norm_path = ""
routes_norm_path= ""
sources_norm_path = ""
aggregate_path = ""

weather_info = ""
routes_info = ""
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
  
def get_routes_info():
    routes_dic = {}
    total_lengths = np.array(routes_info['total_length'])
    average_widths = np.array(routes_info['average_width']) 
    intersection_ids =  np.array(routes_info['intersection_id'])
    tollgate_ids = np.array(routes_info['tollgate_id'])
    length = len(total_lengths)
    for i in range(length):
        id = intersection_ids[i]+'-'+str(tollgate_ids[i])
        routes_dic[id] = {}
        routes_dic[id]['total_length'] = total_lengths[i]
        routes_dic[id]['average_width'] = average_widths[i]
    return routes_dic
    
def getweatherarr(isval):
    weatherarr = []
    for c in globalcolumes:
        if isval:
            _,traindic = get_Discrete_Weather(c, 10)
        else:
            traindic,_ = get_Discrete_Weather(c, 10)
        weatherarr.append(traindic)
    return weatherarr 
    
def getweathercoder(weatherarr, date, phase):
    coder = []
    for w in weatherarr:
        for c in w[date][phase]:
            coder.append(c)
    return coder
 
def comparedate(date,keydate):
    if date == keydate:
        return date
    else:
        if '-' in keydate:
            date = date.replace('/','-')
        else:
            date = date.replace('-','/')
        return date
    
def discrete_aggregate(isval):
    weekdayarr = get_Discrete_Normtime(7)
    normtimearr = get_Discrete_Normtime(72)
    totallen = 7+72
    totallen += len(globalcolumes*10)
    weatherarr = getweatherarr(isval)
    columes = []
    for i in range(totallen):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"average_width"', '"total_length"', '"avg_travel_time"'])
    columes = np.hstack((columes, restColumes))
    fw = open(aggregate_path, 'w')
    fw.writelines(','.join(columes) + '\n')
    
    routes_dic = get_routes_info()
    dates = np.array(sources_info['date'])
    avg_travel_times = np.array(sources_info['avg_travel_time']) 
    time_intervals = np.array(sources_info[['time_interval']]) 
    intersection_ids =  np.array(sources_info['intersection_id'])
    tollgate_ids = np.array(sources_info['tollgate_id'])
    keys = list(weatherarr[0].keys())
    length = len(dates)
    for i in range(length):
        info = []
        id = intersection_ids[i]+'-'+str(tollgate_ids[i])
        phase,date = getphase(time_intervals[i], dates[i])
        date = comparedate(date, keys[0])
        if not date in weatherarr[0]:
            continue
        if not phase in weatherarr[0][date]:
            continue
        if isholiday(date):
            continue
        day = getweekday(date)
        weekcoder = weekdayarr[int(day)]
        # print(weekcoder)
        interval = int(time_intervals[i]%72)
        normtimecoder = normtimearr[interval]
        weathercoder = getweathercoder(weatherarr, date, phase)
        coder = np.hstack((weekcoder, normtimecoder,weathercoder))
        for j in range(len(coder)):
            info.append('"' + str(coder[j]) + '"')
        info = np.array(info)
        restinfo = np.array(['"' + str(routes_dic[id]['average_width'])+ '"','"'+str(routes_dic[id]['total_length'])+ '"','"'+str(avg_travel_times[i])+ '"'])
        info = np.hstack((info,restinfo))
        out_line = ','.join(info)+'\n'
        fw.writelines(out_line)  
    fw.close()   
 
def aggregate():
    weather_dic = get_weather_info()
    routes_dic = get_routes_info()
    dates = np.array(sources_info['date'])
    avg_travel_times = np.array(sources_info['avg_travel_time']) 
    # avg_travel_times = zeroNormalize(avg_travel_times)
    
    intersection_ids =  np.array(sources_info['intersection_id'])
    tollgate_ids = np.array(sources_info['tollgate_id'])
    norm_times = np.array(sources_info['norm_time'])
    time_intervals = np.array(sources_info[['time_interval']]) 
    length = len(dates)
    
    # weekdic = getNormalizeWeekday(dates)
    holidaydic = getNormalizeHoliday(dates)
    weekarr = getweekarr(dates)
    # holidayarr = getholidayarr(dates)
    # encodingarr = np.hstack((weekarr, holidayarr))
    # encodingarr = np.hstack((weekarr, time_intervals)) 
    encodingres = encoder(weekarr)
    columes = []
    for i in range(len(encodingres[0])):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"holiday"', '"norm_time"', '"average_width"', '"total_length"', '"pressure"', '"sea_pressure"',
    '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"', '"avg_travel_time"'])
    columes = np.hstack((columes, restColumes))
    # print(columes)
    fw = open(aggregate_path, 'w')
    fw.writelines(','.join(columes) + '\n')
    for i in range(length):
        id = intersection_ids[i]+'-'+str(tollgate_ids[i])
        phase,date = getphase(time_intervals[i], dates[i])
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
        restinfo = np.array(['"' + str(holidaydic[date])+ '"', '"' + str(norm_times[i])+ '"', '"' + str(routes_dic[id]['average_width'])+ '"','"'+str(routes_dic[id]['total_length'])+ '"',
        '"'+str(weather_info[0])+ '"', '"'+str(weather_info[1])+ '"', '"'+str(weather_info[2])+ '"', '"'+str(weather_info[3])+ '"', '"'+str(weather_info[4])+ '"',
        '"'+str(weather_info[5])+ '"', '"'+str(weather_info[6])+ '"', '"'+str(avg_travel_times[i])+ '"'])
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
    global routes_norm_path
    global sources_norm_path
    global aggregate_path

    global weather_info
    global routes_info
    global sources_info
    if not isVal:
        weather_norm_path = pardir+"/dataSets/training/norm_weather (table 7)_training.csv"
        routes_norm_path= pardir+"/dataSets/training/norm_widthandlength.csv"
        sources_norm_path = pardir+"/dataSets/training/norm_avg_travel_time_new.csv"
        aggregate_path = pardir+"/dataSets/training/totaldata.csv"
    else:
        weather_norm_path = pardir+"/dataSets/testing_phase1/norm_weather (table 7)_test1.csv"
        sources_norm_path = pardir+"/dataSet_phase2/train/norm_avg_travel_time_new.csv"
        routes_norm_path= pardir+"/dataSets/training/norm_widthandlength.csv"
        aggregate_path = pardir+"/dataSets/testing_phase1/totaldata.csv"
    
    weather_info = pd.read_csv(weather_norm_path,encoding='utf-8')
    routes_info = pd.read_csv(routes_norm_path,encoding='utf-8')
    sources_info = pd.read_csv(sources_norm_path,encoding='utf-8')
    discrete_aggregate(isVal)  
        
if  __name__ == "__main__":
    aggregate_main(0)