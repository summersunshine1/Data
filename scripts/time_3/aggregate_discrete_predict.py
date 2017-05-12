# -*- coding: utf-8 -*-
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

aggregate_path = pardir+"/dataSets/testing_phase1/predict_discrete_data.csv"

intervals = []
globalcolumes = ["pressure", "wind_direction", "wind_speed", "temperature", "rel_humidity"]

def getweathercoder(weatherarr, date, phase):
    coder = []
    for w in weatherarr:
        for c in w[date][phase]:
            coder.append(c)
    return coder
    
def aggregate():
    weekdayarr = get_Discrete_Normtime(7)
    normtimearr = get_Discrete_Normtime(72)
    totallen = 7+72
    weatherarr = []
    for c in globalcolumes:
        _,testdic = get_Discrete_Weather(c, 10)
        weatherarr.append(testdic)
        totallen += 10
    columes = []
    for i in range(totallen):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"date"','"interval"'])
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
    length = len(datearr)
    
    for i in range(length):
        interval = intervalarr[i]
        date = datearr[i]
        phase,date = getphase(intervalarr[i],datearr[i])
        # print(weatherarr[0].keys())
        formatday = date.replace('/','-')
        if not formatday in weatherarr[0]:
            continue
        if not phase in weatherarr[0][formatday]:
            continue
        day = getweekday(date)
        weekcoder = weekdayarr[int(day)]
        normtimecoder = normtimearr[interval%72]
        weathercoder = getweathercoder(weatherarr, formatday, phase)
        coder = np.hstack((weekcoder, normtimecoder,weathercoder))
        info = []
        for j in range(len(coder)):
            info.append('"' + str(coder[j]) + '"')
        info = np.array(info)
        restinfo = np.array(['"' + date+ '"','"'+str(interval)+ '"'])
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