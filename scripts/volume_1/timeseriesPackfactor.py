import pandas as pd
from pandas import DataFrame
from pandas import concat
from plot_sametime import *
import numpy as np
from sklearn.svm import SVR,LinearSVR
from sklearn.model_selection import cross_val_score,KFold, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
import random

def weighted_average(series, n):
    result = 0.0
    weights = getweight(n)
    for n in range(len(weights)):
        result += series[-n-1] * weights[n]
    return result
    
def getweight(n):
    weights = [0.1]
    for i in range(1,n):
        weights.append(weights[i-1]+0.1)
    weights = np.array(weights)
    ratio = 1.0/np.sum(weights)
    weights = weights*ratio
    return weights
    
def normcolumn(x, n):
    x = np.array(x)
    for i in range(n):
        plt.plot(x[:,i], color = 'red')
        x[:,i] = zeroNormalize(x[:,i])
        # plt.plot(x[:,i],color = 'yellow')
        plt.show()
    return x
    
def getfactor(id):
    resdic,holidaydic = getvolumeinfo()
    times = resdic[id].keys()

    n = 7
    print(len(times))
    for time in times:
        arr = resdic[id][time]
        l = len(arr)
        x = []
        y = []
        count = 0
        for i in range(l-n):
            temparr = arr[i:i+n]
            if id in holidaydic and time in holidaydic[id] and (i+n) in holidaydic[id][time]:
                temparr.append(1)
                count+=1
            else:
                temparr.append(0)
            x.append(temparr)
            y.append(arr[i+n])
        x = DataFrame(x)
        x = x.sample(frac=1)

        x = normcolumn(x, n)
        createmodel(x,y)
               
def createmodel(x, y):
    clf = LinearSVR(epsilon = 0.1, C = 1)
    clf.fit(x, y)   
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    
def predict():
    resdic = getvolumeinfo()
    ids = ['1-0','1-1','2-0','3-0','3-1']
    id = '1-0'
    times = resdic[ids[0]].keys()
    n = 7
    
    for time in times:
        arr = resdic[id][time]
        l = len(arr)
        prev_seq = np.array(arr[0:n])
        predict = []
        ground = []
        for i in range(n,l):
            v = weighted_average(prev_seq, n)
            predict.append(v)
            ground.append(arr[i])
            prev_seq = np.hstack((prev_seq[1:],arr[i]))
        plt.plot(predict, color = 'red')
        plt.plot(ground, color = 'yellow')
        plt.show()
            
        
if __name__=="__main__":
    ids = ['1-0','1-1','2-0','3-0','3-1']
    for id in ids:
        getfactor(id)
        break
    
        
   

    
    
    
    

    

