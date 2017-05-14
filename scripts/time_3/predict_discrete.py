# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from datetime import datetime,timedelta
from getPath import *

pardir = getparentdir()
commonpath = pardir + "/scripts/common"

import sys
sys.path.append(commonpath)
from commonLib import *

data_path = ""
result_path = pardir + "/dataSets/testing_phase1/predict_result.csv"

selected_arr = []
intervals = []
dates = []

sources_norm_path = pardir+"/dataSets/testing_phase1/test1_20min_avg_path_travel_time.csv"
sources_info = ""

def getsourceinfo():
    resdic = {}
    links = sources_info["linkid"]
    dates = sources_info["date"]
    intervals = sources_info["interval"]
    travel_times = sources_info["avg_travel_time"]
    l = len(links)
    for i in range(l):
        if not links[i] in resdic:
            resdic[links[i]] = {}
        if not dates[i] in resdic[links[i]]:
            resdic[links[i]][dates[i]] = {}
        resdic[links[i]][dates[i]][int(intervals[i])] = travel_times[i]
    return resdic

def get_source_data(link, selected_cols):
    data = pd.read_csv(data_path,encoding='utf-8')
    x = data[selected_cols][data['linkid']==link]
    y = data['avg_travel_time'][data['linkid']==link] 
    return x,y
    
def get_test_data(selected_cols):
    data = pd.read_csv(data_path,encoding='utf-8')
    x = data[selected_cols]
    return x

def my_custom_loss_func(ground_truth, predictions):
    return np.mean(np.abs(ground_truth-predictions)/ground_truth)
 
def get_model_path(count):
    rfrpath = pardir+"/scripts/model/rfr"
    svrpath = pardir+"/scripts/model/svr"
    rfrpaths = []
    svrpaths = []
    for i in range(1,count+1):
        rpath = rfrpath + str(i) + '.pkl'
        spath = svrpath + str(i) + '.pkl'
        rfrpaths.append(rpath)
        svrpaths.append(spath)
    return rfrpaths, svrpaths
    
def aggregate_data_by_link(links):
    x_arr = []
    y_arr = []
    for i in range(len(links)):
        link = links[i]
        x,y = get_source_data(link,selected_arr[i])
        x_arr.append(x)
        y_arr.append(y)
    y_arr = np.array(y_arr)
    y = np.sum(y_arr, axis = 0)
    return x,y
    
def getaheadtime():
    resdic = getsourceinfo()
    intervals = np.array([18,19,20,21,22,23,45,46,47,48,49,50])
    dates = np.array(["2016/10/18","2016/10/19","2016/10/20","2016/10/21","2016/10/22","2016/10/23","2016/10/24"])
    links = getlinks()
    newdic = {}
    timearr = []
    
    for link in links:
        if not link in resdic:
            print("no link")
            continue
        if not link in newdic:
            newdic[link] = []
        for date in dates:
            maxcount = 11
            for interval in intervals:
                newinterval = interval+6
                flag = 1
                count = 1
                tempinterval = newinterval-count
                while(flag and count<=maxcount):
                    flag = 0
                    if not tempinterval in resdic[link][date]:
                        flag = 1
                        count+=1
                        tempinterval = newinterval-count
                if flag==1:
                    print("lack")
                    continue
                newdic[link].append(resdic[link][date][tempinterval])
                timearr.append(resdic[link][date][tempinterval])
    std = np.std(timearr)
    avg = np.mean(timearr)
    return newdic,std,avg

def predict(isValidation):
    linkseq = get_link_seperate_path()
    rfrpaths, svrpaths = get_model_path(len(linkseq))
    if isValidation:
        ys = []
        for i in range(len(linkseq)):
            x,y = aggregate_data_by_link(linkseq[i])
            ys.append(y)
            # rf = joblib.load(rfrpaths[i])
            svr = joblib.load(svrpaths[i])
            # predict_y1 = rf.predict(x)
            predict_y2 = svr.predict(x)
            # predict_y = (predict_y1+predict_y1)/2
            print(my_custom_loss_func(y,predict_y2))
                    
    else:
        resdic,std,avg = getaheadtime()
        predict_ys = []
        for i in range(len(linkseq)):
            # rf = joblib.load(rfrpaths[i])
            totaltime = []
            for link in linkseq[i]:
                time = np.array(resdic[link])
                time = (time-avg)/std
                totaltime.append(time)
            totaltime = np.sum(totaltime, axis = 0)
            totaltime = [[t] for t in totaltime]
            x = get_test_data(selected_arr[i])
            x = np.hstack((x,totaltime))
            svr = joblib.load(svrpaths[i])
            # predict_y1 = rf.predict(x)
            predict_y2 = svr.predict(x)
            # predict_y = (predict_y1+predict_y1)/2
            predict_ys.append(predict_y2)
        return predict_ys
        
def aggregate_result(predict_ys):
    ids = get_intersection_toll()
    dic = get_path()
    final_ys = []
    for id in ids:
        seqs = dic[id]
        y = []
        for seq in seqs:
            y.append(predict_ys[seq])
        y = np.array(y)
        final_ys.append(np.sum(y,axis=0))
    return final_ys

def format_result(ys):
    print(len(ys[0]))
    fw = open(result_path, 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
    intervals = np.array([24,25,26,27,28,29,51,52,53,54,55,56])
    dates  = ['2016/10/18','2016/10/19','2016/10/20','2016/10/21','2016/10/22','2016/10/23','2016/10/24']
    
    length = len(ys)
    ids = get_intersection_toll()
    for i in range(length):
        id = ids[i]
        y_arr = ys[i]
        idarr = id.split('-')
        j = 0
        for date in dates:
            for interval in intervals:
                start_time_window,end_time_window = get_time_from_interval(date, interval)
                out_line = ','.join(['"' + idarr[0] + '"', '"' + idarr[1] + '"',
                                         '"[' + str(start_time_window) + ',' + str(end_time_window) + ')"',
                                         '"' + str(y_arr[j]) + '"']) + '\n'
                j+=1
                fw.writelines(out_line)
    fw.close()    
 
def predict_path_main(isValidation, arr):
    global data_path
    global selected_arr
    global sources_info
    sources_info = pd.read_csv(sources_norm_path,encoding='utf-8')
    selected_arr = arr
    print(arr)
    if isValidation:
        data_path = pardir + "/dataSets/testing_phase1/discrete_totaldata.csv"
        predict(isValidation)
    else:
        data_path = pardir + "/dataSets/testing_phase1/predict_discrete_data.csv"
        y = predict(isValidation)
        y = aggregate_result(y)
        format_result(y)

if __name__ == "__main__":
    
    arr = [['40', '41', '62', '78', '86', '88', '91', '93', '111', '128'], ['1', '3', '5',
 '15', '80', '85', '89', '93', '120', '123'], ['1', '8', '11', '34', '45', '81',
 '102', '103', '120', '125'], ['1', '5', '31', '80', '83', '101', '102', '114',
'122', '126'], ['1', '2', '5', '89', '91', '100', '101', '102', '114', '128'], [
'1', '4', '11', '20', '84', '90', '95', '98', '103', '111'], ['21', '81', '92',
'94', '95', '101', '114', '115', '121', '124'], ['1', '2', '4', '49', '83', '91'
, '92', '96', '104', '115']] 
    predict_path_main(0,arr)

    