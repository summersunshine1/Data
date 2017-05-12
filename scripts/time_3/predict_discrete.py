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
        predict_ys = []
        for i in range(len(linkseq)):
            # rf = joblib.load(rfrpaths[i])
            x = get_test_data(selected_arr[i])
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
    predict_path_main(0,see)

    