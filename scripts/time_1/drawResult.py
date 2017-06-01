import pandas as pd
from datetime import datetime,timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pylab as plt
from getPath import *

pardir = getparentdir()

path_1 = pardir+"/res/predict_result9-1.csv"
data_1 = pd.read_csv(path_1, encoding='utf-8')

path_2 = pardir+"/res/predict_result10-1.csv"
data_2 = pd.read_csv(path_2, encoding='utf-8')

path_4 = pardir+"/res/predict_result12-1.csv"
data_4 = pd.read_csv(path_4, encoding='utf-8')

inter_ids = ['A','B','C']
toll_ids = [1,2,3]



volumes_1 = data_1['avg_travel_time']
volumes_2 = data_2['avg_travel_time']
volumes_4 = data_4['avg_travel_time']
plt.plot(volumes_1,color='blue')
plt.plot(volumes_2,color = 'red')
plt.plot(volumes_4,color = 'yellow')
plt.show()