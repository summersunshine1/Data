# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# from sklearn.cross_validation import cross_val_score
import seaborn as sns
from sklearn.linear_model import LinearRegression,ElasticNet,ElasticNetCV
from sklearn.model_selection import cross_val_score,KFold, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVR,LinearSVR
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
import warnings
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append("F:/kdd/scripts/common")
from commonLib import *
                        
warnings.filterwarnings("ignore")
data_path = "F:/kdd/dataSets/training/traveltime_totaldata.csv"

def getcols():
    tempcols = []
    for i in range(7):
        tempcols.append(str(i))
    tempcols = np.array(tempcols)
    restcols = np.array(['holiday', 'norm_time', 'pressure','sea_pressure','wind_direction','temperature','rel_humidity','precipitation'])
    cols = np.hstack((tempcols, restcols))
    return cols
    
def getdata():
    data = pd.read_csv(data_path,encoding='utf-8')
    data = data.sample(frac=1)
    return data

def get_source_data(linkid):
    data = getdata()
    cols = getcols()
    X = data[cols][data['linkid']==linkid]
    y = data['avg_travel_time'][data['linkid']==linkid]        
    return X,y
    
def get_smallest_time_arr(timearr):
    min = len(timearr[0])
    minindex = 0
    for i in range(1, len(timearr)):
        if(len(timearr[i]) < min):
            min = len(timearr[i])
            minindex = i
    return timearr[minindex]
    
def selectfeature(link):
    x,y= get_source_data(link) 
    # clf = RFE(RandomForestRegressor(n_estimators=10, random_state = 0),4)
    clf = RandomForestRegressor(n_estimators=10, random_state = 0)
    fit = clf.fit(x,y)
            
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    # for f in range(15):
        # print("%2d) %-*s %f" %(f+1, 30,cols[indices[f]],importance[indices[f]]))
    x_selected = clf.transform(x,threshold = importance[indices[8]])
    x_selected = x
    # print(x_selected.shape)
    print(link)
    creat_model(x_selected, y)
    
def selectfeature1(x,y,index):
    # clf = RFE(RandomForestRegressor(n_estimators=10, random_state = 0),4)
    clf = RandomForestRegressor(n_estimators=10, random_state = 0)
    fit = clf.fit(x,y)
    cols = getcols()       
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    # for f in range(6):
        # print("%2d) %-*s %f" %(f+1, 30,cols[indices[f]],importance[indices[f]]))
    x_selected = clf.transform(x,threshold = importance[indices[5]])
    
    featurenums = x_selected.shape[1]
    feature_dic = {}
    selected_cols=[]
    for f in range(featurenums):
        selected_cols.append(cols[indices[f]])
        # feature_dic[cols[indices[f]]] = 1
    
    # for c in cols:
        # if c in feature_dic:
            # selected_cols.append(c)
    # print(x_selected)
    # print(selected_cols)
    creat_model(x_selected, y, index)
    return selected_cols
     
def creat_model1(x,y):
    
    # rng = np.random.RandomState(31337)
    # kf = KFold(n_splits=10, shuffle=True, random_state=rng)
    # ground = []
    # pred = []
    # xgb_model = xgb.XGBRegressor(n_estimators = 1000, learning_rate=0.05).fit(x,y)
    # score = make_scorer(my_custom_loss_func, greater_is_better=False)
    # scores = -cross_val_score(xgb_model, x, y,cv=10,scoring=score)
    # print(scores)
    
    # for train_index,test_index in kf.split(x):
        # xgb_model = xgb.XGBRegressor(n_estimators = 1000, subsample = 0.8, learning_rate=0.1 ).fit(x[train_index],y[train_index])
        # predictions = xgb_model.predict(x[test_index])
        # actuals = y[test_index]
        # print(my_custom_loss_func(actuals, predictions))
    # xgb_model = xgb.XGBRegressor()
    # clf = 
    
    # clf = LogisticRegressionCV()#cv=10, penalty = 'l2',solver = 'liblinear')
    # print(y)
    # clf.fit(x, y)  
    # score = make_scorer(my_custom_loss_func, greater_is_better=False)
    # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    # print(scores)
    enet = ElasticNetCV(l1_ratio = 0.7, cv=10)
    enet.fit(x, y)  
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(enet, x, y,cv=10,scoring=score)
    print(scores)

def creat_model(x,y,index):
    # tuned_parameters = [{'epsilon':[0.1,0.2,0.3],'C': [1, 5, 10,15]}]
    # sample_leaf_options = [1,5,10,50,100,200,500]
    # gs = GridSearchCV(estimator = SVR(), param_grid = tuned_parameters, cv = 10)
    # gs = gs.fit(x,y)
    # print(gs.best_params_)
    # params = gs.best_params_
   
   
    # joblib.dump(clf, 'F:/kdd/scripts/rf.pkl')
    # clf = RandomForestRegressor(n_estimators=10,oob_score = TRUE,n_jobs = -1,random_state =1)
    # clf.fit(x,y)
    # clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    # clf = SVR(epsilon = params['epsilon'], C = params['C'])
    # clf.fit(x, y)   
    # score = make_scorer(my_custom_loss_func, greater_is_better=False)
    # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    # print(scores)
    # path = 'F:/kdd/scripts/model/svr'+str(index)+'.pkl'
    # joblib.dump(clf, path)
    
    clf = xgb.XGBRegressor(n_estimators = 1000, learning_rate=0.05).fit(x,y)
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    
    
    # tuned_parameters = [{'epsilon':[0.1,0.2,0.3],'C': [1, 5, 10,15]}]
    # sample_leaf_options = [1,5,10,50,100,200,500]
    # gs = GridSearchCV(estimator = LinearSVR(), param_grid = tuned_parameters, cv = 10)
    # gs = gs.fit(x,y)
    # print(gs.best_params_)
    # params = gs.best_params_
   
    # joblib.dump(clf, 'F:/kdd/scripts/rf.pkl')
    # clf = RandomForestRegressor(n_estimators=10,oob_score = TRUE,n_jobs = -1,random_state =1)
    # clf.fit(x,y)
    # clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    # clf = LinearSVR(epsilon = params['epsilon'], C = params['C'])
    # clf.fit(x, y)   
    # score = make_scorer(my_custom_loss_func, greater_is_better=False)
    # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    # print(scores)
    path = 'F:/kdd/scripts/model/rfr'+str(index)+'.pkl'
    joblib.dump(clf, path)
    # clf = RandomForestRegressor(n_estimators = 200, oob_score = True, n_jobs = -1,random_state =50,
                                # max_features = "auto", min_samples_leaf = 10)

    # clf.fit(x,y)
    # score = make_scorer(my_custom_loss_func, greater_is_better=False)
    # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    # print(scores)
    # path = 'F:/kdd/scripts/model/rfr'+str(index)+'.pkl'
    # joblib.dump(clf, path)
    
def create_path_model_main():
    index = 1#represent different links in global_linkseq
    linkseq = get_link_seperate_path()
    select_arr = []
    for links in linkseq:
        y_arr = []
        for link in links:
            x,y = get_source_data(link)
            y_arr.append(y)
        y_arr = np.array(y_arr)
        y = np.sum(y_arr, axis = 0)
        selected_cols = selectfeature1(x,y,index)
        index += 1
        select_arr.append(selected_cols)
    return select_arr
    
if __name__ == "__main__":
    create_path_model_main()
    
    
    



