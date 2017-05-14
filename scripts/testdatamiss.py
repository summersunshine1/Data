import pandas as pd
import numpy as np
from datetime import datetime

sources_path = "F:/kdd/dataSets/testing_phase1/test1_20min_avg_travel_time.csv"
sources_info = pd.read_csv(sources_path,encoding='utf-8')

def getsourceinfo():
    time_windows = np.array(sources_info['time_window'])
    avg_travel_times = np.array(sources_info['avg_travel_time'])
    intersection_ids = np.array(sources_info['intersection_id'])
    tollgate_ids = np.array(sources_info['tollgate_id'])
    length = len(time_windows)
    resdic = {}
    for i in range(length):
        timepair = time_windows[i].split(',')
        endtime = timepair[1]
        starttime = timepair[0]
        trace_endtime = datetime.strptime(endtime, "%Y-%m-%d %H:%M:%S)")
        trace_starttime = datetime.strptime(starttime, "[%Y-%m-%d %H:%M:%S")
        date = str(trace_starttime.year)+'/'+str(trace_starttime.month)+'/'+str(trace_starttime.day)
        num = (trace_starttime.hour*60+trace_starttime.minute)/20
        id = intersection_ids[i]+'-'+str(tollgate_ids[i])
        if not id in resdic:
            resdic[id] = {}
        if not date in resdic[id]:
            resdic[id][date] = {}
        resdic[id][date][num] = avg_travel_times[i]
    return resdic
    
def identify():
    resdic = getsourceinfo()
    ids = ['A-2','A-3','B-3','B-1','C-3','C-1']
    dates = ["2016/10/18","2016/10/19","2016/10/20","2016/10/21","2016/10/22","2016/10/23","2016/10/24"]
    nums = [18,19,20,21,22,23,45,46,47,48,49,50]
    for id in ids:
        for date in dates:
            for num in nums:
                if not id in resdic:
                    print("id lack:"+id)
                    continue
                if not date in resdic[id]:
                    print("lack"+id+" "+date)
                    continue
                if not num in resdic[id][date]:
                    print("lack" + id + " "+ date+" "+str(num))
                    continue
                    
if __name__=="__main__":
    identify()
                