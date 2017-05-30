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
from plot_sametime import *

globalcolumes = ["pressure", "wind_direction", "wind_speed", "temperature", "rel_humidity"]
sources_norm_path = ""
aggregate_path = ""
sources_info = ""
minlen = 7


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
    toll_ids = sources_info["tollgate_id"]
    time_windows = sources_info["time_window"]
    directions = sources_info["direction"]
    volumes = sources_info["volume"]
    l = len(toll_ids)
    for i in range(l):
        id = str(toll_ids[i])+'-'+str(directions[i])
        if not id in resdic:
            resdic[id] = {}
            
        timepair = time_windows[i].split(',')
        endtime = timepair[1]
        starttime = timepair[0]
        trace_endtime = datetime.strptime(endtime, "%Y-%m-%d %H:%M:%S)")
        trace_starttime = datetime.strptime(starttime, "[%Y-%m-%d %H:%M:%S")
        date = str(trace_starttime.year)+'/'+str(trace_starttime.month)+'/'+str(trace_starttime.day)
        if not date in resdic[id]:
            resdic[id][date] = {}
        interval = get_num_from_hour_minute(trace_starttime.hour,trace_starttime.minute)
        resdic[id][date][interval] = volumes[i]
    return resdic
    
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
    ids = ['1-0','1-1','2-0','3-0','3-1']
    columes = []
    columes.append('"id"')
    weekdayarr = get_Discrete_Normtime(7)
    normtimearr = get_Discrete_Normtime(72)
    totallen = 7+72
    totallen += len(globalcolumes*10)
    weatherarr = getweatherarr(isval)
    for i in range(totallen):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"neighbour"','"volume"'])
    columes = np.hstack((columes, restColumes))
    fw = open(aggregate_path, 'w')
    fw.writelines(','.join(columes) + '\n')
    
    dates = np.array(["2016/10/18","2016/10/19","2016/10/20","2016/10/21","2016/10/22","2016/10/23","2016/10/24"])
    length = len(dates)
    totalresdic,testdic = get_dic_from_totaldata()
    for id in ids:
        resdic,true_volume_dic,_,_ = getneighbourlist(id,totalresdic,testdic)
        keys = list(weatherarr[0].keys())
        neigbour_times=""
        predict_times = ""
        if id == '2-0':
            neigbour_times = get_neighbour_window("6:0:0","18:0:0")
            predict_times = get_neighbour_window("8:0:0","20:0:0")    
        else:
            neigbour_times = get_neighbour_window("0:0:0","22:0:0")
            predict_times = get_neighbour_window("2:0:0","0:0:0")
        for i in range(length):
            for j in range(len(predict_times)):
                timeinterval = get_num_from_timestr(predict_times[j])
                phase,date = getphase(timeinterval, dates[i])
                date = comparedate(date, keys[0])
                if not dates[i] in weatherarr[0]:
                    continue
                if not phase in weatherarr[0][date]:
                    continue
                info = [id]
                day = getweekday(date)
                weekcoder = weekdayarr[int(day)]
                # print(weekcoder)
                interval = int(timeinterval%72)
                normtimecoder = normtimearr[interval]
                weathercoder = getweathercoder(weatherarr, date, phase)
                # print(weathercoder)
                coder = np.hstack((weekcoder, normtimecoder,weathercoder))
                # print(len(coder))
                for k in range(len(coder)):
                    info.append('"' + str(coder[k]) + '"')
                info = np.array(info)
                if not predict_times[j] in resdic:
                    continue
                
                restinfo = np.array(['"' + str(resdic[predict_times[j]][i])+ '"','"'+str(true_volume_dic[predict_times[j]][i])+ '"'])
                info = np.hstack((info,restinfo))
                out_line = ','.join(info)+'\n'
                fw.writelines(out_line)  
    fw.close()
 
def get_dic_from_totaldata():
    totalresdic, _ = get_totaldata()
    testdic = {}
    test_holidaydic = {}
    testdic,_ = getTestInfo(testdic, test_holidaydic)
    return totalresdic,testdic
        
def getneighbourlist(newid,totalresdic,testdic):
    neigbour_times=""
    predict_times = ""
    if newid == '2-0':
        neigbour_times = get_neighbour_window("6:0:0","18:0:0")
        predict_times = get_neighbour_window("8:0:0","20:0:0")    
    else:
        neigbour_times = get_neighbour_window("0:0:0","22:0:0")
        predict_times = get_neighbour_window("2:0:0","0:0:0")
    mean_value_dic = {}
    std_value_dic = {}
    times = totalresdic[newid]
    resdic = {}
    true_volume_dic = {}
    if not newid in mean_value_dic:
        mean_value_dic[newid]={}
    if not newid in std_value_dic:
        std_value_dic[newid]={}
    for time in times:
        mean_value_dic[newid][time] = np.mean(np.array(totalresdic[newid][time]))
        std_value_dic[newid][time] = np.std(np.array(totalresdic[newid][time]))
    
    l = len(neigbour_times)
    i = 0
    while(i<=l-6):
        temp = []
        for j in range(i,i+6):
            arr = normalizebymean(testdic[newid][neigbour_times[j]], mean_value_dic[newid][neigbour_times[j]], std_value_dic[newid][neigbour_times[j]])
            arr = testdic[newid][neigbour_times[j]]
            if(len(arr) == minlen):
                temp.append(arr)
        if len(temp)<6:
            i=i+6
            continue
        for j in range(i,i+6):
            if len(testdic[newid][predict_times[j]])<minlen:
                continue
            resdic[predict_times[j]] = np.mean(temp,axis=0)*std_value_dic[newid][predict_times[j]]+mean_value_dic[newid][predict_times[j]]
            # resdic[predict_times[j]] = np.mean(temp,axis=0)#*std_value_dic[newid][predict_times[j]]+mean_value_dic[newid][predict_times[j]]
            # resdic[predict_times[j]] = np.array(temp)
            true_volume_dic[predict_times[j]] = testdic[newid][predict_times[j]]
            # true_volume_dic[predict_times[j]] =  normalizebymean(testdic[newid][predict_times[j]], mean_value_dic[newid][predict_times[j]], std_value_dic[newid][predict_times[j]])
        i= i+6
    return resdic,true_volume_dic,mean_value_dic,std_value_dic
    
def aggregate_main(isVal):
    global sources_norm_path
    global aggregate_path
    
    global sources_info
    
    if not isVal:
        sources_norm_path = pardir+"/dataSets/training/training_20min_avg_volume_new.csv"
        aggregate_path = pardir+"/dataSets/training/discrete_volume_totaldata.csv"
    else:
        # sources_norm_path = pardir+"/dataSets/testing_phase1/test1_20min_avg_path_travel_time.csv"
        sources_norm_path = pardir+"/dataSet_phase2/test/test2_20min_avg_volume.csv"
        aggregate_path = pardir+"/dataSet_phase2/test/discrete_volume_totaldata.csv"
    
    sources_info = pd.read_csv(sources_norm_path,encoding='utf-8')
    aggregate(isVal)
    
if __name__=="__main__":
    aggregate_main(0)
