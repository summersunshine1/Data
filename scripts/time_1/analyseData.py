import numpy as np
import pandas as pd 
import matplotlib.pylab as plt

from getPath import *
pardir = getparentdir()

weather_train_path = pardir+"/dataSet_phase2/train/weather (table 7)_2.csv"

def analyze():
    data = pd.read_csv(weather_train_path, encoding='utf-8')
    cols = list(data.columns.values)
    cols = cols[2:]
    for c in cols:
        plt.hist(data[c])
        plt.show()
        
analyze()