import pandas as pd
from datetime import datetime,timedelta
import numpy as np

data_path = "F:/kdd/dataSets/training/training_20min_avg_volume.csv"
new_data_path = "F:/kdd/dataSets/training/training_20min_avg_volume_update.csv"

data_path = "F:/kdd/dataSets/testing_phase1/test1_20min_avg_volume.csv"
new_data_path = "F:/kdd/dataSets/testing_phase1/test1_20min_avg_volume_update.csv"

data = pd.read_csv(data_path,encoding='utf-8')
time_interval = np.array(data['time_window'])
ids = np.array(data['tollgate_id'])
directions = np.array(data['direction'])
volumes = np.array(data['volume'])

new_ids = []
new_directions = []
new_volumns = []

l = len(time_interval)

changed_times = []
for i in range(l):
    # if ids[i] == 1 and directions[i] == 0:
    
    time_arr = time_interval[i].split(',')
    trace_time = datetime.strptime(time_arr[0], "[%Y-%m-%d %H:%M:%S")
    
    # if trace_time.month==10 and trace_time.day>=1 and trace_time.day<=7:
        # continue
    newtime = trace_time.strftime("%Y-%m-%d %H:%M:%S")
    changed_times.append(newtime)
    new_ids.append(ids[i])
    new_directions.append(directions[i])
    new_volumns.append(volumes[i])
 
l = len(changed_times)
fw = open(new_data_path, 'w')
fw.writelines(','.join(['"tollgate_id"', '"time_window"','"direction"', '"volume"']) + '\n')
for i in range(l):
    out_line = ','.join(['"' + str(new_ids[i]) + '"', '"' + str(changed_times[i]) + '"', '"' + str(new_directions[i]) + '"', '"' + str(new_volumns[i]) + '"']) + '\n'
    fw.writelines(out_line)
fw.close()
    

    