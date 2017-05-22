import pandas as pd
import numpy as np

path1 = 'F:/kdd/res/predicted_volume2-15.csv'
path2 = 'F:/kdd/res/predicted_volume2-11.csv'

def writetofile(path, dic):
    fw = open(path, 'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
    for k,v in dic.items():
        k_arr = k.split('_')
        info = np.array(['"' + str(k_arr[1]) + '"','"' + str(k_arr[2]) + '"', '"' + str(k_arr[0]) + '"', '"' + str(v) + '"'])
        out_line = ','.join(info)+'\n'
        fw.writelines(out_line) 
    fw.close()
    
    

def together():
    data1 = pd.read_csv(path1,encoding='utf-8')
    data2 = pd.read_csv(path2,encoding='utf-8')
    
    direction1= data1["direction"]
    toll1 = data1["tollgate_id"]
    time1 = data1["time_window"]
    avg_travel_time1 = data1["volume"]
    
    direction2 = data2["direction"]
    toll2 = data2["tollgate_id"]
    time2 = data2["time_window"]
    avg_travel_time2 = data2["volume"]
    
    dic1 = {}
    dic2 = {}
    
    length1 = len(direction1)
    length2 = len(direction2)
    
    for i in range(length1):
        id = str(direction1[i])+'_'+str(toll1[i])+'_'+time1[i]
        dic1[id] = avg_travel_time1[i]
    for i in range(length2):
        id = str(direction2[i])+'_'+str(toll2[i])+'_'+time2[i]
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
    writetofile('F:/kdd/res/predicted_volume2-16.csv',dic)
    
        
    
    
    