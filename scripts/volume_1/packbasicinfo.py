import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pylab as plt
import functools
import dtw

from getPath import *
pardir = getparentdir()
common_path = pardir+'/scripts/common'
knnwaroppath = pardir+"/scripts"
import sys
sys.path.append(common_path)
sys.path.append(knnwaroppath)
from commonLib import *
from plot_sametime import *
from knntimewarp import *

def cmp_datetime(a, b):
    tempa = datetime.strptime(a[0],"%H:%M:%S")
    tempb = datetime.strptime(b[0],"%H:%M:%S")
    if tempa>tempb: 
        return 1
    elif tempa<tempb:
        return -1
    else:
        return 0
    
def getbasicinfo():
    resdic,holidaydic = getvolumeinfo()
    ids = ['1-0','1-1','2-0','3-0','3-1']

    for id in ids:
        times = resdic[id]
        for time in times:
            resdic[id][time] = np.log(resdic[id][time])
      
    predict_times = []
    for id in ids:
        times = resdic[id]
        times = sorted(times.items(),key = functools.cmp_to_key(cmp_datetime))
        keys = []
        values = []
        for i in range(len(times)):
            keys.append(times[i][0])
            values.append(times[i][1])
        m = KnnDtw(n_neighbors=3)
        l = len(times)
        arr = []
        for i in range(1,l):
            arr.append(np.array(values[i]))
        arr = np.array(arr)
        m.fit(arr)
        y = np.array([values[0]])
        knn_idx,dm = m.regress(y)
        print(id+" "+keys[0])
        arr = []
        fig = plt.figure()
        ax = plt.subplot(121)
        ax2 = plt.subplot(122)
        time = getfollowingtime(keys[0])
        for i in range(len(time)):
            ax.plot(resdic[id][time[i]],label = time[i])
            print(keys[i+1]+" "+str(dm[i+1]))
        ax2.plot(values[0],label = keys[0])
        for i in knn_idx[0]:
            ax2.plot(values[i],label = keys[i])
            print(keys[i]+" "+str(dm[i]))
        ax.legend()
        ax2.legend()
        plt.show()

def getnearneigbours():
    ids = ['1-0','1-1','2-0','3-0','3-1']
    finalresultdic = {}
    newresdic,holidaydic = getvolumeinfo()
    totalresdic, _ = addTestInfo(newresdic,holidaydic)
    mean_value_dic = {}
    std_value_dic = {}
    for id in ids:
        times = totalresdic[id]
        if not id in mean_value_dic:
            mean_value_dic[id]={}
        if not id in std_value_dic:
            std_value_dic[id]={}
        for time in times:
            mean_value_dic[id][time] = np.mean(np.array(totalresdic[id][time]))
            std_value_dic[id][time] = np.std(np.array(totalresdic[id][time]))

    testdic,_ = addTestInfo()
    resdic={}
    for id in ids:
        times = newresdic[id]
        if not id in resdic:
            resdic[id]={}
        for time in times:
            resdic[id][time] = normalizebymean(newresdic[id][time], mean_value_dic[id][time], std_value_dic[id][time])
    test_times = getPredicttimes()
    train_times = getPredicttimes("6:0:0","15:0:0")
    for id in ids:
        if not id in finalresultdic:
            finalresultdic[id] = {}
        train_arr = []
        m = KnnDtw(n_neighbors=3)
        for time in train_times:
            train_arr.append(resdic[id][time])
        train_arr = np.array(train_arr)
        m.fit(train_arr)
        for time in test_times:    
            y = np.array([resdic[id][time]])
            knn_idx,dm = m.regress(y)
            temp = []
            for i in knn_idx[0]:
                if not id in testdic:
                    continue
                if not train_times[i] in testdic[id]:
                    continue
                arr = normalizebymean(testdic[id][train_times[i]], mean_value_dic[id][train_times[i]], std_value_dic[id][train_times[i]])
                temp.append(arr)
            temp = np.array(temp)
            finalresultdic[id][time] = np.mean(temp,axis=0)*std_value_dic[id][time]+mean_value_dic[id][time]
                
            # fig = plt.figure()
            # ax = plt.subplot(111)
            # for i in knn_idx[0]:
                # ax.plot(train_arr[i],label = train_times[i])
            # ax.plot(resdic[id][time],label = time) 
            # ax.legend()
            # plt.show()
    return finalresultdic        
     
def writeResTofile(finalresultdic):
    test_times = getPredicttimes()
    final_res_path = pardir+"/dataSets/testing_phase1/predicted_volume1.csv"
    # if not os.path.exists(final_res_path):
        # fw = open(final_res_path,'w')
        # fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"'])+'\n')
    # else:
    fw = open(final_res_path,'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"'])+'\n')   
    dates = np.array(["2016/10/18", "2016/10/19", "2016/10/20", "2016/10/21","2016/10/22", "2016/10/23","2016/10/24"]) 
    ids = ['1-0','1-1','2-0','3-0','3-1']
    for id in ids:
        temparr = id.split('-')
        toll_id = temparr[0]
        direction = temparr[1]
        for i in range(len(dates)):
            for time in test_times:
                start_time_window,end_time_window = get_time_from_datetime(dates[i], time)
                out_line = ','.join(['"' + str(toll_id) + '"', '"[' + str(start_time_window) + ',' + str(end_time_window) + ')"', 
                '"' + str(direction) + '"', '"' + str(int(finalresultdic[id][time][i])) + '"',]) + '\n'
                fw.writelines(out_line)
    fw.close()
    
 
if __name__ == "__main__":
    getbasicinfo()
    # finalresultdic = getnearneigbours()
    # writeResTofile(finalresultdic)
    # file_path = 