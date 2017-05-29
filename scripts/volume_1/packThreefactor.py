import pandas as pd
from datetime import datetime,timedelta
import numpy as np
from sklearn.externals import joblib

from getPath import *
pardir = getparentdir()

from plot_sametime import *

id = ""
direction = ""
neigbour_times = ""
predict_times = ""
predict_dates = ""
minlen = 7

trend_model_path =  pardir+"/dataSets/training/trend.pkl"
seasonal_path =  pardir+"/dataSets/training/season_norm.csv"
weather_path = pardir +"/dataSets/testing_phase1/norm_weather (table 7)_test1.csv"
volume_path = pardir+"/dataSet_phase2/train/training2_20min_avg_volume.csv"
volume_test_path =  pardir+"/dataSets/testing_phase1/predict_voulume_data.csv"

final_path = pardir+"/dataSet_phase2/train/predict_data.csv"

def get_neighbour_window(starttime,endtime):
    time_windows=[]
    start = datetime.strptime(starttime, "%H:%M:%S")
    end = datetime.strptime(endtime, "%H:%M:%S")
    time = start
    timestr = get_str_from_time(time)
    endstr = get_str_from_time(end)
    while(timestr!=endstr):
        time_windows.append(timestr)
        time += timedelta(minutes = 20)
        timestr = get_str_from_time(time)
    return time_windows

def setid_direction(id_, direction_):
    global id
    global direction
    global neigbour_times
    global predict_times
    global predict_dates
    id = id_
    direction = direction_
    newid = str(id)+'-'+str(direction)
    if newid == '2-0':
        neigbour_times = get_neighbour_window("6:0:0","18:0:0")
        predict_times = get_neighbour_window("8:0:0","20:0:0")    
    else:
        neigbour_times = get_neighbour_window("0:0:0","22:0:0")
        predict_times = get_neighbour_window("2:0:0","0:0:0")
    predict_dates = np.array(["2016/10/18","2016/10/19","2016/10/20","2016/10/21","2016/10/22","2016/10/23","2016/10/24"])
    
def getseasonal():
    seasonal = pd.read_csv(seasonal_path, encoding='utf-8')
    seasonal_volume = seasonal['residual'][-72:]
    seasonal_volume = list(seasonal_volume)
    res_dic = {}
    res_dic[72] = seasonal_volume[0]
    for i in range(1,72):
        res_dic[i] = seasonal_volume[i]
    return res_dic
    
# def get_weather_info():
    # weather_info = pd.read_csv(weather_path, encoding='utf-8')
    # weather_dic = {}
    # cols = list(predict_data.columns.values)
    
    # date = np.array(weather_info['date'])
    # hour = np.array(weather_info['hour'])
    # length = len(date)
    # for i in range(length):
        # if not date[i] in weather_dic:
            # weather_dic[date[i]] = {}
        # if not hour[i] in weather_dic[date[i]]:
            # weather_dic[date[i]][hour[i]]
        # for col in cols:
            # weather_dic[date[i]][hour[i]].append(weather_info[col][i])
    # return weather_dic,date,hour
    
def model_predict(model_path,cols):
    predict_data = pd.read_csv(volume_test_path, encoding='utf-8')
    x = predict_data[cols]
    clf = joblib.load(model_path)
    predict_y = clf.predict(x)
    return predict_y 

    
def getneighbour():
    newid = str(id)+'-'+str(direction)
    newresdic,holidaydic = getvolumeinfo()
    totalresdic, _ = addTestInfo(newresdic,holidaydic)
    mean_value_dic = {}
    std_value_dic = {}
    times = totalresdic[newid]
    resdic = {}
    true_volume_dic = {}
    if not newid in mean_value_dic:
        mean_value_dic[newid]={}
    if not newid in std_value_dic:
        std_value_dic[newid]={}
    for time in times:
        mean_value_dic[newid][time] = np.mean(np.array(totalresdic[newid][time]))
        std_value_dic[newid][time] = np.std(np.array(totalresdic[newid][time]))        
    testdic,_ = addTestInfo()
    l = len(neigbour_times)
    i = 0
    while(i<=l-6):
        temp = []
        for j in range(i,i+6):
            arr = normalizebymean(testdic[newid][neigbour_times[j]], mean_value_dic[newid][neigbour_times[j]], std_value_dic[newid][neigbour_times[j]])
            # arr = testdic[newid][neigbour_times[j]]
            if(len(arr) == minlen):
                temp.append(arr)
            else:
                if len(arr) > minlen:
                    # print(neigbour_times[j])
                    # print(arr)
                    # print(len(arr))
                    temp.append(arr)

        for j in range(i,i+6):
            resdic[predict_times[j]] = np.mean(temp,axis=0)*std_value_dic[newid][predict_times[j]]+mean_value_dic[newid][predict_times[j]]
            # resdic[predict_times[j]] = np.mean(temp,axis=0)
            true_volume_dic[predict_times[j]] = testdic[newid][predict_times[j]]
        i= i+6
    return resdic,true_volume_dic
    
def getneighbourlist():
    newid = str(id)+'-'+str(direction)
    newresdic,holidaydic = getvolumeinfo()
    totalresdic, _ = addTestInfo(newresdic,holidaydic)
    mean_value_dic = {}
    std_value_dic = {}
    times = totalresdic[newid]
    resdic = {}
    true_volume_dic = {}
    if not newid in mean_value_dic:
        mean_value_dic[newid]={}
    if not newid in std_value_dic:
        std_value_dic[newid]={}
    for time in times:
        mean_value_dic[newid][time] = np.mean(np.array(totalresdic[newid][time]))
        std_value_dic[newid][time] = np.std(np.array(totalresdic[newid][time]))        
    testdic,_ = addTestInfo()
    l = len(neigbour_times)
    i = 0
    while(i<=l-6):
        temp = []
        for j in range(i,i+6):
            # arr = normalizebymean(testdic[newid][neigbour_times[j]], mean_value_dic[newid][neigbour_times[j]], std_value_dic[newid][neigbour_times[j]])
            arr = testdic[newid][neigbour_times[j]]
            if(len(arr) == minlen):
                temp.append(arr)
        if len(temp)<6:
            i=i+6
            continue
        for j in range(i,i+6):
            # resdic[predict_times[j]] = np.mean(temp,axis=0)*std_value_dic[newid][predict_times[j]]+mean_value_dic[newid][predict_times[j]]
            resdic[predict_times[j]] = np.array(temp)
            true_volume_dic[predict_times[j]] = testdic[newid][predict_times[j]]
        i= i+6
    return resdic,true_volume_dic
    
def writetofile(path,cols):
    fw = open(path, 'w')
    columes = []
    for i in range(6):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(['"date"', '"time"', '"season"', '"trend"','"volume"'])
    columes = np.hstack((columes, restColumes))
    # columes = np.array(['"date"', '"time"', '"neighbour"', '"season"', '"trend"','"volume"'])
    fw.writelines(','.join(columes) + '\n')
    
    seasondic = getseasonal()
    resdic,true_volume_dic= getneighbourlist()
    predict_res =  model_predict(trend_model_path, cols)
    c = 0
    
    for i in range(len(predict_dates)):
        for time in predict_times:
            num = get_num_from_timestr(time)
            season = seasondic[num]
            trend = predict_res[c]
            # neighbour = resdic[time][i]
            if not time in resdic:
                continue
            neighbour = resdic[time][:,i]
            if len(true_volume_dic[time])<minlen:
                continue
            volume = true_volume_dic[time][i]
            info = neighbour
            restinfo = np.array(['"' + predict_dates[i] + '"', '"' + time + '"', '"' + str(season) + '"','"' + str(trend) + '"','"' + str(volume) + '"']) 
            info = np.hstack((info, restinfo))
            # info = np.array(['"' + predict_dates[i] + '"', '"' + time + '"','"' + str(neighbour) + '"', '"' + str(season) + '"','"' + str(trend) + '"','"' + str(volume) + '"'])
            out_line = ','.join(info)+'\n'    
            fw.writelines(out_line)
            c+=1
    fw.close()
    
def packThreefactor_main(id_, direction_,cols):
    setid_direction(id_, direction_)
    writetofile(final_path, cols)

def testdatalack():
    data = pd.read_csv(volume_path, encoding='utf-8')
    columes = data.columns.values
    print(columes[0])
    counts = data.groupby('time_window').size()
    index = counts.index
    # for i in range(len(counts)):
        # if(counts[i]<5):
            # print(index[i])
    

if __name__=="__main__":
    packThreefactor_main([])
    