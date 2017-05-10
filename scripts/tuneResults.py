import pandas as pd
from datetime import datetime,timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pylab as plt

final_res_path = "F:/kdd/res/predicted_volume2-3.csv"
volume_path_2 = "F:/kdd/res/predicted_volume2-2.csv"
data = pd.read_csv(volume_path_2, encoding='utf-8')
tollgate_ids = data['tollgate_id']
time_windows = data['time_window']
directions = data['direction']
volumes = data['volume']
l = len(tollgate_ids)
fw = open(final_res_path,'w')
fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"'])+'\n')
for i in range(l):
    timepair = time_windows[i].split(',')
    endtime = timepair[1]
    starttime = timepair[0]
    trace_starttime = datetime.strptime(starttime, "[%Y-%m-%d %H:%M:%S")
    starttime = trace_starttime + timedelta(minutes=20)
    endtime = starttime + timedelta(minutes=20)
    out_line = ','.join(['"' + str(tollgate_ids[i]) + '"', '"[' + str(starttime) + ',' + str(endtime) + ')"', 
        '"' + str(directions[i]) + '"', '"' + str(int(volumes[i])) + '"',]) + '\n'
    fw.writelines(out_line)
fw.close()
        
