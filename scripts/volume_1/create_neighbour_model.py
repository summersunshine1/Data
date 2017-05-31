import pandas as pd
from numpy import *
import numpy as np
# from sklearn.cross_validation import cross_val_score
# import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.svm import SVR,LinearSVR
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

from getPath import *
pardir = getparentdir()
datapath = pardir + '/dataSets/training/discrete_volume_totaldata.csv'
commonpath = pardir+'/scripts/common'
import sys
sys.path.append(commonpath)
from commonLib import *

model_path = pardir+'/scripts/model/neigbourmodel.pkl'

def create_model():
    data = pd.read_csv(datapath,encoding='utf-8')
    cols = list(data.columns.values)
    # cols = cols[1:-2]
    cols = cols[1:-2]
    ids = ['1-0','1-1','2-0','3-0','3-1']
    data = data.sample(frac=1)
    # for id in ids:
        # print(id)
    x = np.array(data[cols])#[data["id"]==id])
    # x1 = data['a'][data["id"]==id]
    # x2 = data['b'][data["id"]==id]
    # x3 = data['c'][data["id"]==id]
    # x4 = data['d'][data["id"]==id]
    # x5 = data['e'][data["id"]==id]
    # x6 = data['f'][data["id"]==id]
    # x1 = zeroNormalize(x1)
    # x1 = np.array([[t] for t in x1])
    # x2 = zeroNormalize(x2)
    # x2 = np.array([[t] for t in x2])
    # x3 = zeroNormalize(x3)
    # x3 = np.array([[t] for t in x3])
    # x4 = zeroNormalize(x4)
    # x4 = np.array([[t] for t in x4])
    # x5 = zeroNormalize(x5)
    # x5 = np.array([[t] for t in x5])
    # x6 = zeroNormalize(x6)
    # x6 = np.array([[t] for t in x6])
    y = data['volume']#[data["id"]==id]    
    x1 = data['neighbour']#[data["id"]==id]
    y1= [t for t in y]
    x11 = [t for t in x1]
    # plt.plot(x11)
    # plt.plot(y1)
    # plt.show()
    # x1 = zeroNormalize(x1)
    x1 = np.array([[t] for t in x1])
    x = np.hstack((x,x1))
    pca = PCA(n_components=len(cols)+1)
    pca.fit(x)
    feature_arr = pca.components_
    print(feature_arr.shape)
    trysize = [10,20,30,40,50,60,70,80,90,100,110,120,130]
    results = []
    minscore = 100
    minsize = -1
    for size in trysize:
        part = feature_arr[:,:size]
        temp = mat(x)*mat(part)
        clf = LinearSVR(C=1, epsilon=0.1)
        clf.fit(temp, y) 
        score = make_scorer(my_custom_loss_func, greater_is_better=False)
        scores = -cross_val_score(clf, temp, y,cv=10,scoring=score)
        print(scores)
        mean_score = np.mean(scores)
        if mean_score < minscore:
            minscore = meanscore
            minsize = size
        print(np.mean(scores))
        results.append(np.mean(scores))
    return minsize
    # plt.plot(results)
    # plt.show()
    
    # x =  np.hstack((x,x1,x2,x3,x4,x5,x6))
    # clf = LinearSVR(C=1, epsilon=0.1)
    # clf = RandomForestRegressor(n_estimators = 100, n_jobs = -1,random_state =50,
                                # max_features = "auto", min_samples_leaf = 1)
    # clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)

    # plt.plot(x1,color = 'r')
    # plt.plot(y)
    # plt.show()
    # clf.fit(x, y)   
    # score = make_scorer(my_custom_loss_func, greater_is_better=False)
    # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    # joblib.dump(clf, model_path) 
    # print(scores)
    # print(np.mean(scores))
    
if __name__=="__main__":
    create_model()