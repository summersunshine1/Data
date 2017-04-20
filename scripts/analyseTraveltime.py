import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from datetime import datetime

sources_path = "F:/kdd/dataSets/training/training_20min_avg_travel_time.csv"
sources_info = pd.read_csv(sources_path,encoding='utf-8')
time_windows = sources_info['time_window'][(sources_info['intersection_id']=='B')&(sources_info['tollgate_id'] == 3)]
avg_travel_times = sources_info['avg_travel_time'][(sources_info['intersection_id']=='B')&(sources_info['tollgate_id'] == 3)]
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
avg_travel_times = avg_travel_times-avg_travel_times.shift(72)
v2 = pd.DataFrame(avg_travel_times) 
# t = pd.DatetimeIndex(datetimearr)
# v2.index = t
plt.plot(v2)
plt.show()

