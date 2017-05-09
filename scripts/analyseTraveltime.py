import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from datetime import datetime
from knntimewarp import *

intersections = ['A','B','C']
tollgates = [1,2,3]

sources_path = "F:/kdd/dataSets/training/training_20min_avg_travel_time.csv"
sources_info = pd.read_csv(sources_path,encoding='utf-8')

def getdata(intersection, tollgate):
    time_windows = sources_info['time_window'][(sources_info['intersection_id']==intersection)&(sources_info['tollgate_id'] == tollgate)]
    avg_travel_times = sources_info['avg_travel_time'][(sources_info['intersection_id']==intersection)&(sources_info['tollgate_id'] == tollgate)]
    time_windows = [time for time in time_windows]
    avg_travel_times = [time for time in avg_travel_times]
    return time_windows, avg_travel_times
    
def getdatearr(time_windows): 
    length = len(time_windows)
    timearr = []
    dates = []
    datetimearr = []
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
        datetimearr.append(trace_starttime)
    # avg_travel_times = zeroNormalize(avg_travel_times)
    timearr = np.array(timearr)
    # normtimearr = zeroNormalize(timearr)
    dates = np.array(dates)
    datetimearr = np.array(datetimearr)
    return timearr, dates, datetimearr
    
def plot(avg_travel_times, datetimearr):
    # print(len(avg_travel_times))
    # print(len(datetimearr))
    v2 = pd.DataFrame(avg_travel_times) 
    t = pd.DatetimeIndex(datetimearr)
    v2.index = t
    plt.plot(v2)
    #plt.show()
    
def analyze():
    for intersection in intersections:
        for toll in tollgates:
            if intersection == 'A' and toll == 1:
                continue
            if intersection == 'B' and toll == 2:
                continue
            if intersection == 'C' and toll == 2:
                continue
            time_windows, avg_travel_time =  getdata(intersection, toll)
            timearr, dates, datetimearr = getdatearr(time_windows)
            # print(intersection + " "+str(toll)+" "+str(np.mean(avg_travel_time)))
            plot(avg_travel_time, datetimearr)
        plt.show()
            
def getdistance():
   
    for intersectionA in intersections:
        for tollA in tollgates:
            if intersectionA == 'A' and tollA == 1:
                continue
            if intersectionA == 'B' and tollA == 2:
                continue
            if intersectionA == 'C' and tollA == 2:
                continue
            time_windows, avg_travel_timesA = getdata(intersectionA, tollA)
            for intersectionB in intersections:
                for tollB in tollgates:
                    if intersectionB == 'A' and tollB == 1:
                        continue
                    if intersectionB == 'B' and tollB == 2:
                        continue
                    if intersectionB == 'C' and tollB == 2:
                        continue
                    time_windows, avg_travel_timesB = getdata(intersectionB, tollB)
                    m = KnnDtw()
                    # print(avg_travel_timesA[:5])
                    # print(avg_travel_timesB[:5])
                    distance = m._dtw_distance(avg_travel_timesA, avg_travel_timesB)
               
                    print(intersectionA+"-"+str(tollA)+" "+intersectionB+"-"+str(tollB)+str(distance))
                    
getdistance()                   
    
    

