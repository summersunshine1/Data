# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# from sklearn.cross_validation import cross_val_score
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
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
    clf = RandomForestRegressor(n_estimators=10, random_state = 0)
    clf.fit(x,y)
    importance = clf.feature_importances_
    # print(importance)
    indices = np.argsort(importance)[::-1]
    # print(indices)
    for f in range(17):
        print("%2d) %-*s %f" %(f+1, 30,cols[indices[f]],importance[indices[f]]))
    x_selected = clf.transform(x,threshold = 0.0095)
    print(x_selected.shape)
    creat_model(x_selected, y)
    
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
    
    
    



