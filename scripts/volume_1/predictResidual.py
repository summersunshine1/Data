import pandas as pd
import matplotlib.pylab as plt
from pandas import Series
from pandas import DataFrame
from pandas import concat
from statsmodels.tsa.ar_model import AR
import numpy as np
from getPath import *

pardir = getparentdir()

ids = [1,2,3]
directions = [0,1]

def gettrain_error(id, direction):
    residual_path = pardir+"/dataSets/training/error_residual.csv"
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    # residual= pd.read_csv(residual_path, parse_dates=['time_window'], index_col='time_window',date_parser=dateparse)
    residual = pd.read_csv(residual_path, encoding = 'utf-8')
    data = residual['residual'][(residual['id']==id)&(residual['direction'] == direction)]
    data = [d for d in data]
    finalres = []
    for i in range(14):#every period is six and total period is 14*6
        train_resid = data[i*6:i*6+6]
        train_resid = [r for r in train_resid]
        history = train_resid
        predictions = []
        for j in range(6):
            mean = np.mean(history)
            predictions.append(mean)
            history = np.hstack((history[1:],mean))
            finalres.append(mean)
    plt.plot(data, color = 'red')
    plt.plot(finalres, color = 'blue')
    plt.show()
    return finalres
 
if __name__ == "__main__":              
    for id in ids:
        for direct in directions:
            if id == 2 and direct == 1:
                continue
            gettrain_error(id, direct)        
    
     