import pandas as pd
import numpy as np
from datetime import datetime,timedelta

real_volume_path = "F:/kdd/dataSets/testing_phase1/test1_20min_avg_volume_update.csv"
real_data = pd.read_csv(real_volume_path, encoding='utf-8')
id = 3
direction = 1

lstm_predict_path = "F:/kdd/dataSets/training/lstmResult.csv"
predict_data = pd.read_csv(lstm_predict_path, encoding='utf-8')


def gettruevolume():
    time_windows = real_data['time_window'][(real_data['tollgate_id']==id)&(real_data['direction'] == direction)]
    # print(time_windows)
    volumes = real_data['volume'][(real_data['tollgate_id']==id)&(real_data['direction'] == direction)]
    l = len(time_windows)
    time_windows = np.array(time_windows)
    volumes = np.array(volumes)
    
    res_dic = {}
    for i in range(l):
        time = time_windows[i]
        trace_time = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
        date = str(trace_time.year)+'/'+str(trace_time.month)+'/'+str(trace_time.day)
        num = (trace_time.hour*60+trace_time.minute)/20
        if num==0:
            num = 72
        if not date in res_dic:
            res_dic[date]={}
        res_dic[date][num] = volumes[i]
    return res_dic
    
def formPredictandGround():
    res_dic = gettruevolume()
    dates = predict_data['date']
    intervals = predict_data['interval']
    volumes = predict_data['volume']
    l = len(dates)
    predicted_data = []
    ground_data = []
    for i in range(l):
        if not dates[i] in res_dic:
            continue
        if not intervals[i] in res_dic[dates[i]]:
            continue
        ground_data.append(res_dic[dates[i]][intervals[i]])
        predicted_data.append(volumes[i])
    predicted_data = np.array(predicted_data)
    ground_data = np.array(ground_data)
    return predicted_data,ground_data
    
def my_custom_loss_func(ground_truth, predictions):
    return np.mean(np.abs(ground_truth-predictions)/ground_truth)
  
if __name__=="__main__":
    predicted_data,ground_data = formPredictandGround()
    print(my_custom_loss_func(ground_data[0:72], predicted_data[0:72]))