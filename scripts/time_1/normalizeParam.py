# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
from getPath import *
pardir = getparentdir()
commonpath = pardir + "/scripts/common"
import sys
sys.path.append(commonpath)
from commonLib import *
# from sklearn import preprocessing

weather_path = pardir + "/dataSets/training/weather (table 7)_training.csv"
routes_path = pardir + "/dataSets/training/widthandlength.csv"
sources_path = pardir + "/dataSets/training/training_20min_avg_travel_time.csv"


weather_norm_path = pardir + "/dataSets/training/norm_weather (table 7)_training.csv"
routes_norm_path= pardir + "/dataSets/training/norm_widthandlength.csv"
sources_norm_path = pardir + "/dataSets/training/norm_avg_travel_time.csv"

weather_path = pardir + "/dataSets/testing_phase1/weather (table 7)_test1.csv"
sources_path = pardir + "/dataSets/testing_phase1/test1_20min_avg_travel_time.csv"

weather_norm_path = "F:/kdd/dataSets/testing_phase1/norm_weather (table 7)_test1.csv"
sources_norm_path = "F:/kdd/dataSets/testing_phase1/norm_avg_travel_time.csv"

sources_path = pardir + "/dataSets/training/training_20min_avg_travel_time_new.csv"
sources_norm_path = pardir +"/dataSets/training/norm_avg_travel_time_new.csv"

# sources_path = pardir + "/dataSet_phase2/test/test2_20min_avg_travel_time.csv"
# sources_norm_path = pardir +"/dataSet_phase2/train/norm_avg_travel_time_new.csv"

weather_info = pd.read_csv(weather_path,encoding='utf-8')
routes_info = pd.read_csv(routes_path,encoding='utf-8')
sources_info = pd.read_csv(sources_path,encoding='utf-8')
    
def NormWinddirection(arr):
    newarr = []
    l = len(arr)
    replaceset = []
    for i in range(l):
        if not arr[i]==999017:
            newarr.append(arr[i])
        else:
            print(i)
            replaceset.append(i)
    newarr = np.array(newarr)
    mu = np.average(newarr)
    for i in replaceset:
        arr[i] = mu    
    sigma = np.std(arr)
    return (arr-mu)/sigma
            

def NormalizeWeather():
    pressures = np.array(weather_info['pressure'])
    sea_pressures = np.array(weather_info['sea_pressure'])
    wind_directions = np.array(weather_info['wind_direction'])
    wind_speeds = np.array(weather_info['wind_speed'])
    temperatures = np.array(weather_info['temperature'])
    rel_humiditys = np.array(weather_info['rel_humidity'])
    precipitations = np.array(weather_info['precipitation'])
    pressures = zeroNormalize(pressures)
    sea_pressures = zeroNormalize(sea_pressures)
    wind_directions = NormWinddirection(wind_directions)
    wind_speeds = zeroNormalize(wind_speeds)
    temperatures = zeroNormalize(temperatures)
    rel_humiditys = zeroNormalize(rel_humiditys)
    precipitations = zeroNormalize(precipitations)
    return pressures,sea_pressures,wind_directions,wind_speeds,temperatures,rel_humiditys,precipitations
    
def writeWeatherNorm(filepath):
    pressures,sea_pressures,wind_directions,wind_speeds,temperatures,rel_humiditys,precipitations = NormalizeWeather()
    fw = open(filepath, 'w')
    fw.writelines(','.join(['"date"', '"hour"', '"pressure"', '"sea_pressure"', '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"']) + '\n')
    dates = np.array(weather_info['date'])
    hours = np.array(weather_info['hour'])
    length = len(dates)
    for i in range(length):
        if '-' in dates[i]:
            dates[i] = dates[i].replace('-', '/')
        out_line = ','.join(['"' + dates[i] + '"', '"' + str(hours[i]) + '"',
                                 '"'+str(pressures[i])+ '"','"'+str(sea_pressures[i])+ '"','"'+str(wind_directions[i])+ '"','"'+str(wind_speeds[i])+ '"','"'+str(temperatures[i])+ '"','"'+str(rel_humiditys[i])+ '"',
                                 '"' + str(precipitations[i]) + '"']) + '\n'
        fw.writelines(out_line)  
    fw.close()
    
def NormalizeRoutes():
    total_lengths = np.array(routes_info['total_length'])
    average_widths = np.array(routes_info['average_width']) 
    total_lengths = zeroNormalize(total_lengths)
    average_widths = zeroNormalize(average_widths) 
    return total_lengths,average_widths
    
def writeRoutesNorm(filepath):
    total_lengths,average_widths = NormalizeRoutes()
    fw = open(filepath, 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"total_length"', '"average_width"']) + '\n')
    intersection_ids =  np.array(routes_info['intersection_id'])
    tollgate_ids = np.array(routes_info['tollgate_id'])
    length = len(intersection_ids)
    for i in range(length):
        out_line = ','.join(['"' + intersection_ids[i] + '"', '"' + str(tollgate_ids[i])+ '"','"'+str(total_lengths[i])+ '"','"'+str(average_widths[i])+ '"']) + '\n'
        fw.writelines(out_line) 
    fw.close()

def NormalizeSources():
    time_windows = np.array(sources_info['time_window'])
    avg_travel_times = np.array(sources_info['avg_travel_time']) 
    length = len(time_windows)
    timearr = []
    dates = []
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
        # date = date.strftime("%Y/%m/%d")
        dates.append(date)
    # avg_travel_times = zeroNormalize(avg_travel_times)
    timearr = np.array(timearr)
    normtimearr = zeroNormalize(timearr)
    dates = np.array(dates)
    return timearr,normtimearr,avg_travel_times,dates
        
def writeSourcesNorm(filepath):
    timearr,normtimearr,avg_travel_times,dates= NormalizeSources()
    fw = open(filepath, 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"date"', '"norm_time"', '"time_interval"', '"avg_travel_time"']) + '\n')
    intersection_ids =  np.array(sources_info['intersection_id'])
    tollgate_ids = np.array(sources_info['tollgate_id'])
    length = len(intersection_ids)
    for i in range(length):
        out_line = ','.join(['"' + intersection_ids[i] + '"', '"' + str(tollgate_ids[i])+ '"','"' +str(dates[i])+ '"','"'+str(normtimearr[i])+ '"', '"'+str(timearr[i])+ '"','"'+str(avg_travel_times[i])+ '"']) + '\n'
        fw.writelines(out_line) 
    fw.close()        
            
    
if __name__ == "__main__":
    # writeWeatherNorm(weather_norm_path)
    # writeRoutesNorm(routes_norm_path)
    writeSourcesNorm(sources_norm_path)
                                 
                                 
                                 
                                 
                             
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 