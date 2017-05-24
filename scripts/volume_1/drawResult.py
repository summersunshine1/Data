import pandas as pd
from datetime import datetime,timedelta
import numpy as np
# from statsmodels.tsa.stattools import adfuller
import matplotlib.pylab as plt
from getPath import *
pardir = getparentdir()


volume_path_1 = pardir+"/res/predicted_volume2-15.csv"
volume_data_1 = pd.read_csv(volume_path_1, encoding='utf-8')

volume_path_2 = pardir+"/res/predicted_volume2-11.csv"
volume_data_2 = pd.read_csv(volume_path_2, encoding='utf-8')

volume_path_4 = pardir+"/res/predicted_volume2-17.csv"
volume_data_4 = pd.read_csv(volume_path_4, encoding='utf-8')

ids = [1,2,3]
directions = [0,1]

for id in ids:
    for direction in directions:
        if id == 2 and direction == 1:
            continue
        volumes_1 = volume_data_1['volume'][(volume_data_1['tollgate_id']==id)&(volume_data_1['direction'] == direction)]
        volumes_2 = volume_data_2['volume'][(volume_data_2['tollgate_id']==id)&(volume_data_2['direction'] == direction)]
        volumes_4 = volume_data_4['volume'][(volume_data_4['tollgate_id']==id)&(volume_data_4['direction'] == direction)]
        plt.plot(volumes_1,color='blue')
        plt.plot(volumes_2,color = 'red')
        plt.plot(volumes_4,color = 'yellow')
        plt.show()