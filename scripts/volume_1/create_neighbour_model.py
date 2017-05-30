import pandas as pd
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

from getPath import *
pardir = getparentdir()
datapath = pardir + '/dataSets/training/discrete_volume_totaldata.csv'
commonpath = pardir+'/scripts/common'
import sys
sys.path.append(commonpath)
from commonLib import *

def create_model():
    data = pd.read_csv(datapath,encoding='utf-8')
    cols = list(data.columns.values)
    cols = cols[1:-2]
    x = np.array(data[cols])
    x1 = data['neighbour']
    x1 = zeroNormalize(x1)
    x1 = np.array([[t] for t in x1])
    x =  np.hstack((x,x1))
    clf = LinearSVR(C=1, epsilon=0.1)
    y = data['volume']
    # clf = RandomForestRegressor(n_estimators = 100, n_jobs = -1,random_state =50,
                                    # max_features = "auto", min_samples_leaf = 1)
    clf.fit(x, y)   
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    print(np.mean(scores))
    
if __name__=="__main__":
    create_model()