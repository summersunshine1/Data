# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

weather_path = 'E:/kdd/Data/dataSets/training/weather (table 7)_training.csv'
update_path = 'E:/kdd/Data/dataSets/training/weather (table 7)_training_update.csv'
def readfile(path):
    data = pd.read_csv(path,encoding='utf-8')
    return data
    
def detectOutlier(data):
    names = data.columns.values 
    for i in range(2, len(names)):
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data[[names[i]]])
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        print(labels)
        print(centers)
        
def updateoutlier(data):
    wind_direction = data['wind_direction']
    directions = []
    for d in wind_direction:
        if not d == 999017:
            directions.append(d)
    directions = [d for d in directions]
    average = np.mean(directions)
    average = round(average,2)
    data.loc[data['wind_direction'] == 999017, 'wind_direction'] = average
    l = len(data.columns.values)
    # for i in range(len(directions)):
        # if directions[i] == 999017:
            # data['wind_direction'][i] = average
    # print(data[list(range(0,l))])
    data.to_csv(update_path, index = False)
    
    
if __name__ == "__main__":
    data = readfile(weather_path)
    # detectOutlier(data)
    updateoutlier(data)
    
    
    
    