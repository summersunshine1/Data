# -*- coding: utf-8 -*-
#!/usr/bin/env python

file_suffix = '.csv'
path = ""  # set the data directory
import math
from datetime import datetime,timedelta
import sys
sys.path.append("F:/kdd/scripts/common")
from commonLib import *
from handlePath import *
   
def avgTravelTime(in_file):
    out_suffix = '_20min_avg_path_travel_time'
    in_file_name = in_file + file_suffix
    out_file_name = path + in_file.split('_')[1] + out_suffix + file_suffix
    id_linkseq_dic = getRoutesDic()
    # Step 1: Load trajectories
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()

    # Step 2: Create a dictionary to store travel time for each route per time window
    travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
    for i in range(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]
        id = str(intersection_id)+'-'+str(tollgate_id)
        linkseq = id_linkseq_dic[id]
        trace_start_time = each_traj[3]
        trace_start_time = trace_start_time.replace("-","/")
        temparr = trace_start_time.split(":")
        if(len(temparr)==2):
            trace_start_time = datetime.strptime(trace_start_time, "%Y/%m/%d %H:%M")
        else:
            trace_start_time = datetime.strptime(trace_start_time, "%Y/%m/%d %H:%M:%S")
        time_window_minute = math.floor(trace_start_time.minute / 20) * 20
        start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)
        # start_time_window = str(trace_start_time.year)+'/'+str(trace_start_time.month)+'/'+str(trace_start_time.day)+' '+ str(trace_start_time.hour)+':'+str(trace_start_time.minute)+':00'
        travel_seq = each_traj[4]
        seqs = travel_seq.split(';')
        templink = []
        for seq in seqs:
            details = seq.split('#')
            linkid = int(details[0])
            templink.append(linkid)
        if not len(linkseq) == len(templink):#if templink lack some link
            print("lack"+ str(i))
            continue
   
        for seq in seqs:
            details = seq.split('#')
            linkid = int(details[0])
            time = details[2]
            if not linkid in travel_times:
                travel_times[linkid] = {}
            if start_time_window in travel_times[linkid]:
                travel_times[linkid][start_time_window].append(float(time))
            else:
                travel_times[linkid][start_time_window] = [float(time)]
                
    route_time_windows1 = list(travel_times[113].keys())
    route_time_windows2 = list(travel_times[106].keys())
    print(list(set(route_time_windows2) - set(route_time_windows1)))
    print(list(set(route_time_windows1) - set(route_time_windows2)))

    # Step 3: Calculate average travel time for each route per time window
    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['"linkid"', '"date"','"interval"','"norm_time"', '"avg_travel_time"']) + '\n')
    links = getlinks()
    for linkid in links:
        route_time_windows = list(travel_times[linkid].keys())
        route_time_windows.sort()
        dates,nums = get_date_num_from_timewindow(route_time_windows)
        norm_nums = zeroNormalize(nums)
        for i in range(len(route_time_windows)):
            time_window_start = route_time_windows[i]
            tt_set = travel_times[linkid][time_window_start]
            avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)
            out_line = ','.join(['"' + str(linkid) + '"', '"' +str(dates[i])+ '"' , '"' +str(nums[i])+ '"' ,
                                 '"' +str(norm_nums[i])+ '"' ,'"' + str(avg_tt) + '"']) + '\n'
            fw.writelines(out_line)
    fw.close()

def aggregate_path_main(istest):
    global path
    infile = ""
    if not istest:
        path = 'F:/kdd/dataSets/training/'
        in_file = 'trajectories(table 5)_training_new'
    else:
        # path = 'F:/kdd/dataSets/testing_phase1/'
        # in_file = 'trajectories(table 5)_test1'
        path = 'F:/kdd/dataSet_phase2/test/'
        in_file = 'trajectories(table 5)_test2'
    avgTravelTime(in_file)

if __name__ == '__main__':
    aggregate_path_main(1)