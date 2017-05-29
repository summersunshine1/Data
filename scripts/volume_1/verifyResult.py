import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pylab as plt

from getPath import *
pardir = getparentdir()

commonpath = pardir+'/scripts/common'
import sys
sys.path.append(commonpath)
from commonLib import *

true_data_path = pardir + "/dataSet_phase2/train/training2_20min_avg_volume.csv"
predict_data_path = pardir + "/res/predicted_volume2-15.csv"

def get_data(path):
    times = getPredicttimes()
    data = pd.read_csv(path, encoding='utf-8')
    time_windows = data["time_window"]
    tollgate_ids = data["tollgate_id"]
    directions = data["direction"]
    volumes = data["volume"]
    resdic = {}
    length = len(volumes)
    for i in range(length):
        id = str(tollgate_ids[i])+'-'+str(directions[i])
        if not id in resdic:
            resdic[id] = []
        each_pass = time_windows[i].split(',')
        hourtime = (each_pass[0].split())[1]
        hourtime = get_timestr_from_str(hourtime)
        if hourtime in times:
            resdic[id].append(volumes[i])
            
    return resdic
    
    
def compare_result():
    true_dic = get_data(true_data_path)
    predict_dic = get_data(predict_data_path)
    ids = list(true_dic.keys())
    ground = []
    predict = []
    for id in ids:
        plt.plot(true_dic[id],color='r')
        plt.plot(predict_dic[id])
        plt.title(id)
        plt.show()
        ground = np.hstack((ground, true_dic[id]))
        predict = np.hstack((predict, predict_dic[id]))
        print(id)
        print(my_custom_loss_func(np.array(true_dic[id]),np.array(predict_dic[id])))
    print(my_custom_loss_func(ground, predict))   
if __name__ == "__main__":
    compare_result()
    
        
        
        
    