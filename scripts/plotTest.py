import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
import sys

weather_path = "dataSets/training/weather (table 7)_training_update.csv"

def getPath():
	pwd = sys.path[0]
	index = pwd.rfind('\\')
	path = pwd[:index+1]+weather_path
	return path

def readData(path):
    data = pd.read_csv(path,encoding='utf-8')
    df = data
    # print(data.describe())
    # print(data['rel_humidity'].unique())
    # print(data['rel_humidity'])
    # plt.hist(df['precipitation'], bins = 10)#, range =(df['rel_humidity'].min(),df['rel_humidity'].max()))
    names = data.columns.values
    for i in range(2,len(names)):
        plt.hist(df[names[i]], bins = 5)#, range =(df['rel_humidity'].min(),df['rel_humidity'].max()))
        plt.show()
        # for j in range(i+1, len(names)):
            # title = names[i]+'-'+names[j]
            # plt.scatter(data[names[i]],data[names[j]])
            # plt.title(title)
            # plt.show()
        # arr = [a for a in range(10)]
        # print(pd.cut(df[names[i]],10, labels = arr))
    
if __name__ == "__main__":
	path = getPath()
	readData(path)	