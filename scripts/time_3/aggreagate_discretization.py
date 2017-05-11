import pandas as pd
import numpy as np
import math
from datetime import datetime
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

def aggregate():
    links = getlinks()
    columes = []
    columes.append('"linkid"')
    weekdayarr = get_Discrete_Normtime(7)
    normtimearr = get_Discrete_Normtime(72)
    totallen = 7+72
    weatherarr = []
    for c in globalcolumes:
        traindic,_ = get_Discrete_Weather(c, 10)
        weatherarr.append(traindic)
        totallen += 10  
    for i in range(totallen):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"dateandtime"', '"avg_travel_time"'])
    columes = np.hstack((columes, restColumes))
    fw = open(aggregate_path, 'w')
    fw.writelines(','.join(columes) + '\n')
    for link in links:
        dates = np.array(sources_info['date'][sources_info['linkid']==link])
        avg_travel_times = np.array(sources_info['avg_travel_time'][sources_info['linkid']==link])
        time_intervals = np.array(sources_info['interval'][sources_info['linkid']==link])         
        length = len(dates)
        
        for i in range(length):
            phase,date = getphase(time_intervals[i], dates[i])
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
            dateandtime = dates[i]+"-"+str(time_intervals[i])
            restinfo = np.array(['"' + dateandtime+ '"','"'+str(avg_travel_times[i])+ '"'])
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
        sources_norm_path = pardir+"/dataSets/testing_phase1/test1_20min_avg_path_travel_time.csv"
        aggregate_path = pardir+"/dataSets/testing_phase1/discrete_totaldata.csv"
    
    sources_info = pd.read_csv(sources_norm_path,encoding='utf-8')
    aggregate()
    
if __name__=="__main__":
    aggregate_main(0)
