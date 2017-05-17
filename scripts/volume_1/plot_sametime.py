import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pylab as plt

from getPath import *
pardir = getparentdir()
volume_path = pardir + '/dataSets/training/training_20min_avg_volume.csv'
volume_test_path = pardir + '/dataSets/testing_phase1/test1_20min_avg_volume.csv'
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
        resdic[id][time].append(float(volumes[i]))
    return resdic,holidaydic
    
def addTestInfo(resdic, holidaydic):
    info = pd.read_csv(volume_test_path, encoding='utf-8')
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
        resdic[id][time].append(float(volumes[i]))
    return resdic,holidaydic
    
def get_totaldata():
    resdic,holidaydic = getvolumeinfo()
    resdic,holidaydic = addTestInfo(resdic,holidaydic)
    return resdic,holidaydic
  
def plot(resdic):
    ids = ['1-0','1-1','2-0','3-0','3-1']
    times = resdic[ids[0]].keys()
    for time in times:
        if(len(resdic[ids[0]][time]) == len(resdic[ids[4]][time])):
            plt.plot(resdic[ids[1]][time])
            # plt.plot(resdic[ids[1]][time])
            plt.plot(resdic[ids[4]][time])
            # plt.plot(resdic[ids[4]][time])
            plt.show()
            
if __name__=='__main__':
    resdic = getvolumeinfo()
    plot(resdic)
    
    
    
    
    