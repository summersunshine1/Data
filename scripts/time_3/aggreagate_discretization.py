import pandas as pd
import numpy as np
import math
from datetime import datetime,timedelta
from getPath import *

pardir = getparentdir()
commonpath = pardir + "/scripts/common"

import sys
sys.path.append(commonpath)
from commonLib import *
from handlePath import *
from convertcontinuous import *

globalcolumes = ["pressure", "wind_direction", "wind_speed", "temperature", "rel_humidity"]
sources_norm_path = ""
aggregate_path = ""
sources_info = ""

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

def getsourceinfo():
    resdic = {}
    links = sources_info["linkid"]
    dates = sources_info["date"]
    intervals = sources_info["interval"]
    travel_times = sources_info["avg_travel_time"]
    l = len(links)
    for i in range(l):
        if not links[i] in resdic:
            resdic[links[i]] = {}
        if not dates[i] in resdic[links[i]]:
            resdic[links[i]][dates[i]] = {}
        resdic[links[i]][dates[i]][intervals[i]] = travel_times[i]
    return resdic
    
def getlastdate(date):
    trace_time = datetime.strptime(date, "%Y/%m/%d")
    trace_time = trace_time - timedelta(days=1)
    year = trace_time.year
    month = trace_time.month
    day = trace_time.day
    date = str(year)+"/"+str(month)+"/"+str(day)
    return date
    
def get_last_traveltime():
    resdic = getsourceinfo()
    links = getlinks()
    weatherarr = getweatherarr(0)
    keys = list(weatherarr[0].keys())
    finalres = []
    for link in links:
        dates = np.array(sources_info['date'][sources_info['linkid']==link])
        time_intervals = np.array(sources_info['interval'][sources_info['linkid']==link]) 
        l = len(dates)
        for i in range(l):
            phase,date = getphase(time_intervals[i], dates[i])
            formatdate = comparedate(date, keys[0])
            if not formatdate in weatherarr[0]:
                continue
            if not phase in weatherarr[0][formatdate]:
                continue
            if not link in resdic:
                continue
            maxwindow = 6
            flag = 1
            while(flag and maxwindow>0):
                newdate = date
                flag = 0
                if time_intervals[i]==72:
                    newdate = getlastdate(date)
                    last = time_intervals[i]-maxwindow
                else:
                    last = time_intervals[i]-maxwindow
                    if int(last)== 0:
                        last = 72
                    elif last<0:
                        newdate = getlastdate(date)
                        last = last+72
                if not newdate in resdic[link]:
                    flag = 1
                    maxwindow-=1
                    continue
                if not last in resdic[link][newdate]:
                    flag = 1
                maxwindow-=1
            if flag==1 and maxwindow==0:
                print(str(link)+" "+date + " "+ str(time_intervals[i])+"lack")
                continue
            finalres.append(resdic[link][newdate][last])
    finalres = np.array(finalres)
    mean = np.mean(finalres)
    std = np.std(finalres)
    return mean,std     

def getnew_lasttime():
    resdic = getsourceinfo()
    links = getlinks()
    weatherarr = getweatherarr(0)
    keys = list(weatherarr[0].keys())
    finalres = []
    for link in links:
        dates = np.array(sources_info['date'][sources_info['linkid']==link])
        time_intervals = np.array(sources_info['interval'][sources_info['linkid']==link]) 
        l = len(dates)
        for i in range(l):
            phase,date = getphase(time_intervals[i], dates[i])
            formatdate = comparedate(date, keys[0])
            if not formatdate in weatherarr[0]:
                continue
            if not phase in weatherarr[0][formatdate]:
                continue
            if not link in resdic:
                continue
            maxwindow = 11
            window = 1
            flag = 1
            while(flag and window<=maxwindow):
                newdate = date
                flag = 0
                if time_intervals[i]==72:
                    newdate = getlastdate(date)
                    last = time_intervals[i]-window
                else:
                    last = time_intervals[i]-window
                    if int(last)== 0:
                        last = 72
                    elif last<0:
                        newdate = getlastdate(date)
                        last = last+72
                if not newdate in resdic[link]:
                    flag = 1
                    window+=1
                    continue
                if not last in resdic[link][newdate]:
                    flag = 1
                window+=1
            if flag==1:
                print(str(link)+" "+date + " "+ str(time_intervals[i])+"lack")
                continue
            finalres.append(resdic[link][newdate][last])
    finalres = np.array(finalres)
    mean = np.mean(finalres)
    std = np.std(finalres)
    return mean,std     

    
  
def getweatherarr(isval):
    weatherarr = []
    for c in globalcolumes:
        if isval:
            _,traindic = get_Discrete_Weather(c, 10)
        else:
            traindic,_ = get_Discrete_Weather(c, 10)
        weatherarr.append(traindic)
    return weatherarr      
    
def aggregate(isval):
    resdic = getsourceinfo()
    mean,std = getnew_lasttime()
    links = getlinks()
    columes = []
    columes.append('"linkid"')
    weekdayarr = get_Discrete_Normtime(7)
    normtimearr = get_Discrete_Normtime(72)
    totallen = 7+72
    totallen += len(globalcolumes*10)
    weatherarr = getweatherarr(isval)
    for i in range(totallen):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"dateandtime"','"avg_travel_time"'])
    columes = np.hstack((columes, restColumes))
    fw = open(aggregate_path, 'w')
    fw.writelines(','.join(columes) + '\n')
    for link in links:
        dates = np.array(sources_info['date'][sources_info['linkid']==link])
        avg_travel_times = np.array(sources_info['avg_travel_time'][sources_info['linkid']==link])
        time_intervals = np.array(sources_info['interval'][sources_info['linkid']==link])         
        length = len(dates)
        keys = list(weatherarr[0].keys())
        for i in range(length):
            phase,date = getphase(time_intervals[i], dates[i])
            date = comparedate(date, keys[0])
            if not date in weatherarr[0]:
                continue
            if not phase in weatherarr[0][date]:
                continue
            info = [link]
            day = getweekday(date)
            weekcoder = weekdayarr[int(day)]
            # print(weekcoder)
            interval = int(time_intervals[i]%72)
            normtimecoder = normtimearr[interval]
            weathercoder = getweathercoder(weatherarr, date, phase)
            # print(weathercoder)
            coder = np.hstack((weekcoder, normtimecoder,weathercoder))
            # print(len(coder))
            for j in range(len(coder)):
                info.append('"' + str(coder[j]) + '"')
            info = np.array(info)
            
            # maxwindow = 11
            # window = 1
            # flag = 1
            # while(flag and window<=maxwindow):
                # newdate = date
                # flag = 0
                # if time_intervals[i]==72:
                    # newdate = getlastdate(date)
                    # last = time_intervals[i]-window
                # else:
                    # last = time_intervals[i]-window
                    # if int(last)== 0:
                        # last = 72
                    # elif last<0:
                        # newdate = getlastdate(date)
                        # last = last+72
                # if not newdate in resdic[link]:
                    # flag = 1
                    # window+=1
                    # continue
                # if not last in resdic[link][newdate]:
                    # flag = 1
                # window+=1
            # if flag==1:
                # continue
            # temp = (resdic[link][newdate][last]-mean)/std
            restinfo = np.array(['"' + str(1)+ '"','"'+str(avg_travel_times[i])+ '"'])
            info = np.hstack((info,restinfo))
            out_line = ','.join(info)+'\n'
            fw.writelines(out_line)  
    fw.close()
    
def aggregate_main(isVal):
    global sources_norm_path
    global aggregate_path
    
    global sources_info
    
    if not isVal:
        sources_norm_path = pardir+"/dataSets/training/training_20min_avg_path_travel_time.csv"
        aggregate_path = pardir+"/dataSets/training/discrete_totaldata.csv"
    else:
        # sources_norm_path = pardir+"/dataSets/testing_phase1/test1_20min_avg_path_travel_time.csv"
        sources_norm_path = pardir+"/dataSet_phase2/test/test2_20min_avg_path_travel_time.csv"
        aggregate_path = pardir+"/dataSets/testing_phase1/discrete_totaldata.csv"
    
    sources_info = pd.read_csv(sources_norm_path,encoding='utf-8')
    aggregate(isVal)
    
if __name__=="__main__":
    aggregate_main(0)
