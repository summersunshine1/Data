import pandas as pd
from datetime import datetime,timedelta
from numpy import *
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.externals import joblib
from sklearn.decomposition import PCA

from getPath import *
pardir = getparentdir()
commonpath = pardir+'/scripts/common'
import sys
sys.path.append(commonpath)
from commonLib import *

model_path = pardir+'/scripts/model/neigbourmodel.pkl'
data_path = pardir+'/dataSet_phase2/test/discrete_volume_totaldata.csv'
volume_final_path =  pardir+"/dataSet_phase2/test/predict_voulume_data.csv"

predict_info = pd.read_csv(data_path, encoding = 'utf-8')

# def model_predict(model_path, cols, meanx,stdx): 
def model_predict(model_path, cols, features,mean_x1,std_x1):
    if len(cols)==0:
        cols = list(predict_info.columns.values)
        cols = cols[1:-2]
    temp = cols[:-1]
    x = np.array(predict_info[temp])
   
    newx1 = []
    for t in  predict_info[cols[-1]]:
        newx1.append((t-mean_x1)/std_x1)
    x1 = [[t] for t in newx1]    
    
    # x1 = [[t] for t in predict_info[cols[-1]]] 
    x = np.hstack((x,x1))
    x = mat(x)*mat(features)
    
    # if not minsize==0:
        # pca = PCA(n_components=len(cols))
        # pca.fit(x)
        # feature_arr = pca.components_
        # part = feature_arr[:,:minsize]
        # x = mat(x)*mat(part)
        # print(minsize)
    
    clf = joblib.load(model_path)
    predict_y = clf.predict(x)
    
    x11 = [t for t in predict_info[cols[-1]]]
    y1 = [t for t in predict_y]
    plt.plot(x11,color = 'r')
    plt.plot(y1)
    plt.show()
    return predict_y
    
# def predict_by_new_factor_main(cols,meanx,stdx):
def predict_by_new_factor_main(cols,features,mean_x1,std_x1):
    # predict_y = model_predict(model_path, cols, meanx,stdx)
    predict_y = model_predict(model_path, cols, features,mean_x1,std_x1)
    dates = predict_info['date']
    times = predict_info['time']
    ids = predict_info['id']
    length = len(dates)
    fw = open(volume_final_path,'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"'])+'\n')
    for i in range(length):
        id = ids[i].split('-')
        toll_id = id[0]
        direction = id[1]
        start_time_window,end_time_window = get_time_from_datetime(dates[i], times[i])
        out_line = ','.join(['"' + str(toll_id) + '"', '"[' + str(start_time_window) + ',' + str(end_time_window) + ')"', 
        '"' + str(direction) + '"', '"' + str(int(predict_y[i])) + '"',]) + '\n'
        fw.writelines(out_line)
    fw.close()
    
if __name__=="__main__":
    predict_by_new_factor_main([],0)
        
        
    
    