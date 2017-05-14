import pandas as pd
import matplotlib.pylab as plt
from commonLib import *
from knntimewarp import *
import math

path = "F:/kdd/dataSets/training/training_20min_avg_path_travel_time.csv"
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
data = pd.read_csv(path, parse_dates=['time_window'], index_col='time_window',date_parser=dateparse)

links = getlinks()
# data = pd.read_csv(path, encoding='utf-8')

def getlinkdata():
    linkdata_arr = []
    for link in links:
        link_data = data['avg_travel_time'][(data['linkid']==link)]
        linkdata_arr.append(link_data)
    # print(linkdata_arr)
    return linkdata_arr
 
def analyse(linkdata_arr):
    m = KnnDtw()
    for i in range(len(linkdata_arr)):
        min =  float("inf") 
        minindex  = i 
        for j in range(len(linkdata_arr)):
            if j==i:
                continue
            distance = m._dtw_distance(linkdata_arr[i], linkdata_arr[j])
            if distance<min :
                min = distance
                minindex = j
        print(str(links[i])+"-"+str(links[minindex])+": " +str(min))

if __name__ == "__main__":
    linkdata_arr = getlinkdata()
    analyse(linkdata_arr)
    

    

