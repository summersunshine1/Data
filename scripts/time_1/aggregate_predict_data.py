# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from datetime import datetime
import sys
sys.path.append("F:/kdd/scripts/common")
from commonLib import *

weather_norm_path = "F:/kdd/dataSets/testing_phase1/norm_weather (table 7)_test1.csv"
routes_norm_path= "F:/kdd/dataSets/training/norm_widthandlength.csv"
aggregate_path = "F:/kdd/dataSets/testing_phase1/predict_data.csv"

weather_info = pd.read_csv(weather_norm_path,encoding='utf-8')
routes_info = pd.read_csv(routes_norm_path,encoding='utf-8')

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
    
def aggregate():
    norm_time_dic = getnormtime(intervals)
    weather_dic = get_weather_info()
    routes_dic = get_routes_info()
    
    dates = np.array(["2016/10/18","2016/10/19","2016/10/20","2016/10/21","2016/10/22","2016/10/23","2016/10/24"])
    ids = np.array(['A-2','A-3','B-1','B-3','C-1','C-3'])
    # weekdic = getNormalizeWeekday(dates)
    # holidaydic = getNormalizeHoliday(dates)
    datearr = []
    intervalarr = []
    for id in ids:
        for date in dates:
            for interval in intervals:
                datearr.append(date)
                intervalarr.append(interval)
                
    weekarr = getweekarr(datearr)
    holidayarr = getholidayarr(datearr)
    encodingarr = np.hstack((weekarr, holidayarr))
    tempintervals = np.expand_dims(intervalarr, axis=1)   
    encodingarr = np.hstack((encodingarr, tempintervals)) 
    # print(encodingarr)
    encodingres = encoder(weekarr)
    columes = []
    # for i in range(len(encodingres[0])):
        # columes.append('"' + str(i) + '"')
    columes.append('"' + str(1) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"id"', '"date"','"interval"','"norm_time"','"average_width"', '"total_length"', '"pressure"', '"sea_pressure"',
    '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"'])
    columes = np.hstack((columes, restColumes))
    
    fw = open(aggregate_path, 'w')
    fw.writelines(','.join(columes)+'\n')
    # fw.writelines(','.join(['"id"', '"date"', '"interval"','"norm_time"','"holiday"', 
    # '"week1"', '"week2"','"week3"','"week4"','"week5"','"week6"','"week7"',
    # '"average_width"', '"total_length"', '"pressure"', '"sea_pressure"',
    # '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"']) + '\n')
    
    for id in ids:
        for date in dates:
            for interval in intervals:
                phase,date = getphase(interval,date)
                if not date in weather_dic:
                    continue
                if not phase in weather_dic[date]:
                    continue
                # weekarr = getweekarr(date)
                # weather_info = weather_dic[date][phase]
                # out_line = ','.join(['"' + id + '"', '"' + date + '"', '"' + str(interval) + '"', '"' + str(norm_time_dic[interval]) + '"', '"' + str(isholiday(date)) + '"', '"' + str(getweekday(date)) + '"',
                # '"' + str(routes_dic[id]['average_width'])+ '"','"'+str(routes_dic[id]['total_length'])+ '"',
                # '"'+str(weather_info[0])+ '"', '"'+str(weather_info[1])+ '"', '"'+str(weather_info[2])+ '"', '"'+str(weather_info[3])+ '"', '"'+str(weather_info[4])+ '"',
                # '"'+str(weather_info[5])+ '"', '"'+str(weather_info[6])+ '"']) + '\n'
                
                # coder = encodingres[i]
                weather_info = weather_dic[date][phase]
                info = []
                # for j in range(len(coder)):
                    # info.append('"' + str(coder[j]) + '"')
                info = np.array(info)
                restinfo = np.array(['"' + str(isfixedWeekDay(date, 1)) + '"','"' + id + '"', '"' + date + '"','"' + str(interval) + '"','"' + str(norm_time_dic[interval]) + '"','"' + str(routes_dic[id]['average_width'])+ '"','"'+str(routes_dic[id]['total_length'])+ '"',
                '"'+str(weather_info[0])+ '"', '"'+str(weather_info[1])+ '"', '"'+str(weather_info[2])+ '"', '"'+str(weather_info[3])+ '"', '"'+str(weather_info[4])+ '"',
                '"'+str(weather_info[5])+ '"', '"'+str(weather_info[6])+ '"'])
                info = np.hstack((info,restinfo))
                out_line = ','.join(info)+'\n'
                
                fw.writelines(out_line)  
    fw.close()
    
def aggregate_predict_main(isValidation):
    global intervals
    if isValidation:
        intervals = np.array([18,19,20,21,22,23,45,46,47,48,49,50])
    else:
        intervals = np.array([24,25,26,27,28,29,51,52,53,54,55,56])
    aggregate()
        
if  __name__ == "__main__":
    aggregate_predict_main(0)