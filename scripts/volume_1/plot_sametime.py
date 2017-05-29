import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pylab as plt
import functools

from getPath import *
pardir = getparentdir()
volume_path = pardir + '/dataSets/training/training_20min_avg_volume_new.csv'
volume_test_path = pardir + '/dataSets/testing_phase1/test1_20min_avg_volume.csv'
volume_test_path = pardir+"/dataSet_phase2/train/training2_20min_avg_volume.csv"
# volume_test_path = pardir+"/dataSet_phase2/test/test2_20min_avg_volume.csv"

common_path = pardir+'/scripts/common'

import sys
sys.path.append(common_path)
from commonLib import *

def getvolumeinfo():
    info = pd.read_csv(volume_path, encoding='utf-8')
    resdic = {}
    holidaydic = {}
    tollgate_ids = info["tollgate_id"]
    directions = info["direction"]
    volumes = info["volume"]
    time_windows = info["time_window"]
    l = len(tollgate_ids)
    
    for i in range(l):
        id = str(tollgate_ids[i])+'-'+str(directions[i])
        if not id in resdic:
            resdic[id] = {}
        timepair = time_windows[i].split(',')
        starttime = timepair[0]
        trace_starttime = datetime.strptime(starttime, "[%Y-%m-%d %H:%M:%S")
        time = str(trace_starttime.hour)+':'+str(trace_starttime.minute)+':'+str(trace_starttime.second)
        date = str(trace_starttime.year)+'/'+str(trace_starttime.month)+'/'+str(trace_starttime.day)
        if not time in resdic[id]:
            resdic[id][time] = []
        length = len(resdic[id][time])
        if isholiday(date):
            if not id in holidaydic:
                holidaydic[id] = {} 
            if not time in holidaydic[id]:
                holidaydic[id][time]={}
            holidaydic[id][time][length] = 1
        else:
            resdic[id][time].append(float(volumes[i]))
        
    return resdic,holidaydic
       
def getnewvolumeinfo():
    info = pd.read_csv(volume_path, encoding='utf-8')
    resdic = {}
    holidaydic = {}
    tollgate_ids = info["tollgate_id"]
    directions = info["direction"]
    volumes = info["volume"]
    time_windows = info["time_window"]
    l = len(tollgate_ids)
    
    for i in range(l):
        id = str(tollgate_ids[i])+'-'+str(directions[i])
        if not id in resdic:
            resdic[id] = {}
        timepair = time_windows[i].split(',')
        starttime = timepair[0]
        trace_starttime = datetime.strptime(starttime, "[%Y-%m-%d %H:%M:%S")
        time = str(trace_starttime.hour)+':'+str(trace_starttime.minute)+':'+str(trace_starttime.second)
        date = str(trace_starttime.year)+'/'+str(trace_starttime.month)+'/'+str(trace_starttime.day)
        if not time in resdic[id]:
            resdic[id][time] = {}
        
        resdic[id][time][date] = float(volumes[i])
    return resdic
    
def addTestInfo(resdic={},holidaydic={}):
    resdic = {}
    info = pd.read_csv(volume_test_path, encoding='utf-8')
    tollgate_ids = info["tollgate_id"]
    directions = info["direction"]
    volumes = info["volume"]
    time_windows = info["time_window"]
    l = len(tollgate_ids)
    newresdic ={}
    for i in range(l):
        id = str(tollgate_ids[i])+'-'+str(directions[i])
        if not id in resdic:
            resdic[id] = {}
        timepair = time_windows[i].split(',')
        starttime = timepair[0]
        trace_starttime = datetime.strptime(starttime, "[%Y-%m-%d %H:%M:%S")
        time = str(trace_starttime.hour)+':'+str(trace_starttime.minute)+':'+str(trace_starttime.second)
        date = str(trace_starttime.year)+'/'+str(trace_starttime.month)+'/'+str(trace_starttime.day)
        if not time in resdic[id]:
            resdic[id][time] = []
        length = len(resdic[id][time])
        if isholiday(date):
            print(date)
            if not id in holidaydic:
                holidaydic[id] = {} 
            if not time in holidaydic[id]:
                holidaydic[id][time]={}
            holidaydic[id][time][length] = 1
        else:
            resdic[id][time].append(float(volumes[i]))
    for k,v in resdic.items():
        if not k in newresdic:
            newresdic[k]={}
        for k1,v1 in v.items():
            newresdic[k][k1]=v1
    return newresdic,holidaydic
    
def get_totaldata():
    resdic,holidaydic = getvolumeinfo()
    resdic,holidaydic = addTestInfo(resdic,holidaydic)
    return resdic,holidaydic
 
def cmp_datetime(a, b):
    tempa = datetime.strptime(a[0],"%Y/%m/%d")
    tempb = datetime.strptime(b[0],"%Y/%m/%d")
    if tempa>tempb: 
        return 1
    elif tempa<tempb:
        return -1
    else:
        return 0
        
def cmp_time(a, b):
    tempa = datetime.strptime(a,"%H:%M:%S")
    tempb = datetime.strptime(b,"%H:%M:%S")
    if tempa>tempb: 
        return 1
    elif tempa<tempb:
        return -1
    else:
        return 0
        
def plot(resdic):
    ids = ['1-0','1-1','2-0','3-0','3-1']
    for id in ids:
        times = list(resdic[id].keys())
        times = sorted(times, key = functools.cmp_to_key(cmp_time))
        l = len(times)
        index = 0
        for time in times:
            temps = resdic[id][time]
            temps = sorted(temps.items(),key = functools.cmp_to_key(cmp_datetime))
            keys = []
            values = []
            for i in range(len(temps)):
                keys.append(temps[i][0])
                values.append(temps[i][1])
            values = zeroNormalize(values)
            dates = keys
            v = pd.DataFrame(values)
            dates = get_datetime_from_timearr(dates)
            datesindex = pd.DatetimeIndex(dates)
            v.index = datesindex
            # v = v.resample('W', label='left', closed='left') 
            plt.plot(v)
            if not index==0 and index%6==1:
                # plt.title(id+" "+time)
                # if (index>l-18 and index<l-6):
                    # v = v.shift(-1)
                plt.show()
            index+=1
        # plt.title(id)
        # plt.show()
            
if __name__=='__main__':
    resdic = getnewvolumeinfo()
    plot(resdic)
    
    
    
    
    