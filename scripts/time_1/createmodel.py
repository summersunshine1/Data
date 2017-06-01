# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,KFold, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVR,LinearSVR
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from pandas import read_csv
from sklearn.feature_selection import RFE

from getPath import *
pardir = getparentdir()
commonpath = pardir+'/scripts/common'
import sys
sys.path.append(commonpath)
from commonLib import *
                       
data_path = pardir+"/dataSets/training/totaldata.csv"

def get_source_data():
    data = pd.read_csv(data_path,encoding='utf-8')
    data = data.sample(frac=1)
    cols = list(data.columns.values)
    x = data[cols[:-1]]
    
    y = data[cols[-1]]
    return x,y,cols
    
def my_custom_loss_func(ground_truth, predictions):
    return np.mean(np.abs(ground_truth-predictions)/ground_truth) 
    
def creatmodel_main():
    x,y,cols = get_source_data() 
    clf = LinearSVR(C=1.0, epsilon=0.1)
    clf.fit(x, y)
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    print(np.mean(scores))
    joblib.dump(clf, pardir+'/scripts/svr.pkl')
    # creat_model(x,y)
    
def selectfeature():
    x,y,cols = get_source_data() 
    # print(x)
    # clf = RFE(RandomForestRegressor(n_estimators=10, random_state = 0),4)
    clf = RandomForestRegressor(n_estimators=10, random_state = 0)
    fit = clf.fit(x,y)
    # support_ = fit.support_
    # newcols = []
    # for i in range(len(support_)):
        # if(support_[i]):
            # newcols.append(cols[i])
    # x_selected = x[newcols]
    # print(x_selected)
            
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    # for f in range(17):
        # print("%2d) %-*s %f" %(f+1, 30,cols[indices[f]],importance[indices[f]]))
    # x_selected = clf.transform(x,threshold = importance[indices[5]])
    # print(x_selected.shape)
    creat_model(x, y)
 
def creat_model(x,y):
    # clf = RandomForestRegressor(n_estimators = 100, oob_score = True, n_jobs = -1,random_state =50,
                                # max_features = "auto", min_samples_leaf = 10)

    # clf.fit(x,y)
    # score = make_scorer(my_custom_loss_func, greater_is_better=False)
    # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    # print(scores)
    # print(np.mean(scores))
    # joblib.dump(clf, pardir+'/scripts/rf.pkl')
    # clf = RandomForestRegressor(n_estimators=10,oob_score = TRUE,n_jobs = -1,random_state =1)
    # clf.fit(x,y)
    # clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    clf = LinearSVR(C=1.0, epsilon=0.1)
    # clf = AdaBoostRegressor(n_estimators=100, base_estimator=rg,learning_rate=1)
    clf.fit(x, y)   
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    print(np.mean(scores))
    joblib.dump(clf, pardir+'/scripts/svr.pkl')
    clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    clf.fit(x,y)
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    print(np.mean(scores))
    joblib.dump(clf, pardir+'/scripts/gbt.pkl')
     
     
if __name__ == "__main__":
    creatmodel_main()
    # selectfeature()
    
    
    



