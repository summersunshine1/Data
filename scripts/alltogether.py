import pandas as pd
import numpy as np
import matplotlib.pylab as plt

path1 = 'F:/kdd/res/predict_result2-1.csv'
path2 = 'F:/kdd/res/predict_result10-1.csv'

def writetofile(path, dic):
    fw = open(path, 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
    for k,v in dic.items():
        k_arr = k.split('_')
        info = np.array(['"' + str(k_arr[0]) + '"','"' + str(k_arr[1]) + '"', '"' + str(k_arr[2]) + '"', '"' + str(v) + '"'])
        out_line = ','.join(info)+'\n'
        fw.writelines(out_line) 
    fw.close()
    
    

def together():
    data1 = pd.read_csv(path1,encoding='utf-8')
    data2 = pd.read_csv(path2,encoding='utf-8')
    
    intersection1 = data1["intersection_id"]
    toll1 = data1["tollgate_id"]
    time1 = data1["time_window"]
    avg_travel_time1 = data1["avg_travel_time"]
    
    intersection2 = data2["intersection_id"]
    toll2 = data2["tollgate_id"]
    time2 = data2["time_window"]
    avg_travel_time2 = data2["avg_travel_time"]
    plt.plot(avg_travel_time1)
    plt.plot(avg_travel_time2)
    plt.show()
    
    
    dic1 = {}
    dic2 = {}
    
    length1 = len(intersection1)
    length2 = len(intersection2)
    
    for i in range(length1):
        id = intersection1[i]+'_'+str(toll1[i])+'_'+time1[i]
        dic1[id] = avg_travel_time1[i]
    for i in range(length2):
        id = intersection2[i]+'_'+str(toll2[i])+'_'+time2[i]
        dic2[id] = avg_travel_time2[i] 
    if len(dic1)<len(dic2):
        for k,v in dic1.items():
            dic1[k] += dic2[k]
            dic1[k] = dic1[k]/2
        return dic1
    else:
        for k,v in dic2.items():
            dic2[k] += dic1[k]
            dic2[k] = dic2[k]/2
        return dic2
         
            
if __name__=="__main__":
    dic = together()
    writetofile('F:/kdd/res/predict_result11-1.csv',dic)
    
        
    
    
    