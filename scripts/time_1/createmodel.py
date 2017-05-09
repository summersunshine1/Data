# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# from sklearn.cross_validation import cross_val_score
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,KFold, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from pandas import read_csv
from sklearn.feature_selection import RFE
import xgboost as xgb

import sys
sys.path.append("F:/kdd/scripts/common")
from commonLib import *
                        

data_path = "F:/kdd/dataSets/training/totaldata.csv"

def get_source_data():
    data = pd.read_csv(data_path,encoding='utf-8')

    data = data.sample(frac=1)
    
    tempcols = []
    for i in range(7):
        tempcols.append(str(i))
    tempcols = np.array(tempcols)
    restcols = np.array(['holiday', 'norm_time', 'average_width','total_length','pressure','sea_pressure','wind_direction','temperature','rel_humidity','precipitation'])
    cols = np.hstack((tempcols, restcols))
    X = data[cols]
    # X = data[['norm_time','holiday','week', 'average_width','total_length','pressure','sea_pressure','wind_direction','temperature','rel_humidity','precipitation']]
    
    y = data['avg_travel_time']                                              
    return X,y,cols
    
def my_custom_loss_func(ground_truth, predictions):
    return np.mean(np.abs(ground_truth-predictions)/ground_truth) 
    
def creatmodel_main():
    x,y,cols = get_source_data() 
    # linreg = LinearRegression()
    # score = make_scorer(my_custom_loss_func, greater_is_better=False)
    # scores = -cross_val_score(linreg, x, y,cv=10,scoring=score)
    # print(scores)
    clf = SVR(C=1.0, epsilon=0.1)
    # clf = RandomForestRegressor(n_estimators=10)
    # clf.fit(x,y)
    # clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    clf.fit(x, y)
    
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    joblib.dump(clf, 'F:/kdd/scripts/svr.pkl')
    
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
    
    
def creat_model1(x,y):
    rng = np.random.RandomState(31337)
    kf = KFold(n_splits=10, shuffle=True, random_state=rng)
    ground = []
    pred = []
    xgb_model = xgb.XGBRegressor(n_estimators = 1000, subsample = 0.8, learning_rate=0.5).fit(x,y)
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(xgb_model, x, y,cv=10,scoring=score)
    print(scores)
    
    # for train_index,test_index in kf.split(x):
        # xgb_model = xgb.XGBRegressor(n_estimators = 1000, subsample = 0.8, learning_rate=0.1 ).fit(x[train_index],y[train_index])
        # predictions = xgb_model.predict(x[test_index])
        # actuals = y[test_index]
        # print(my_custom_loss_func(actuals, predictions))
    # xgb_model = xgb.XGBRegressor()
    # clf = 
    

 
def creat_model(x,y):
    # clf = SVR(C=1.0, epsilon=0.1)
    sample_leaf_options = [1,5,10,50,100,200,500]

    # for leaf_size in sample_leaf_options :
    clf = RandomForestRegressor(n_estimators = 200, oob_score = True, n_jobs = -1,random_state =50,
                                max_features = "auto", min_samples_leaf = 10)

    clf.fit(x,y)
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    joblib.dump(clf, 'F:/kdd/scripts/rf.pkl')
    # clf = RandomForestRegressor(n_estimators=10,oob_score = TRUE,n_jobs = -1,random_state =1)
    # clf.fit(x,y)
    # clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    clf = SVR(C=1.0, epsilon=0.1)
    # clf = AdaBoostRegressor(n_estimators=100, base_estimator=rg,learning_rate=1)
    clf.fit(x, y)   
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    joblib.dump(clf, 'F:/kdd/scripts/svr.pkl')
    clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    clf.fit(x,y)
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    joblib.dump(clf, 'F:/kdd/scripts/gbt.pkl')
     
     
if __name__ == "__main__":
    # creatmodel_main()
    selectfeature()
    
    
    



