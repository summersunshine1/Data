# -*- coding: utf-8 -*-
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
# from timeseriescase import *


from getPath import *
pardir = getparentdir()
commonpath = pardir+'/scripts/common'
import sys
sys.path.append(commonpath)
from commonLib import *
from sklearn.metrics import r2_score
import random

residual_path = pardir+"/dataSets/training/residual_norm.csv"
trend_path = pardir+"/dataSets/training/trend_norm.csv"

three_facor_path = pardir+"/dataSet_phase2/train/predict_data.csv"

residual_model_path = pardir+"/dataSets/training/residual.pkl"
trend_model_path = pardir+"/dataSets/training/trend.pkl" 

# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'epsilon':[0.1,0.2,0.3],
                     # 'C': [1, 5, 10, 100]}]

def get_source_data(norm_data_path, isResidual):
    data = pd.read_csv(norm_data_path,encoding='utf-8')
    cols  = list(data.columns.values)
    x = data[cols[:-1]]
    y = data["volume"]
    return x,y,cols
    
def creat_model(source_path, modle_path, isResidual):
    x,y,cols= get_source_data(source_path, isResidual)
    if isResidual:
        clf = SVR(C=1, epsilon=0.1)
    else:
        clf = GradientBoostingRegressor(n_estimators=100)
    clf.fit(x,y)
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    # joblib.dump(clf, modle_path)  
    print(scores)
    
def create_three_factor_model():
    data = pd.read_csv(three_facor_path,encoding='utf-8')
    columes = []
    for i in range(6):
        columes.append('"' + str(i) + '"')
    columes = np.array(columes)
    restColumes = np.array(["season", "trend"])
    columes = np.hstack((columes, restColumes))
    # cols = ["neighbour", "season", "trend"]
    newdata = data.sample(frac=1)
    
    x11 = zeroNormalize(newdata["0"])
    # x12 = zeroNormalize(newdata["1"])
    # x13 = zeroNormalize(newdata["2"])
    # x14 = zeroNormalize(newdata["3"])
    # x15 = zeroNormalize(newdata["4"])
    # x16 = zeroNormalize(newdata["5"])
    # x11 = newdata["0"]
    # x12 = newdata["1"]
    # x13 = newdata["2"]
    # x14 = newdata["3"]
    # x15 = newdata["4"]
    # x16 = newdata["5"]
    x2 = zeroNormalize(newdata["season"])
    x3 = zeroNormalize(newdata["trend"])
    x2 = newdata["season"]
    x3 = newdata["trend"]
    # x1 = np.array([[x] for x in x1])
    x11 = np.array([[x] for x in x11])
    # x12 = np.array([[x] for x in x12])
    # x13 = np.array([[x] for x in x13])
    # x14 = np.array([[x] for x in x14])
    # x15 = np.array([[x] for x in x15])
    # x16 = np.array([[x] for x in x16])
    x2 = np.array([[x] for x in x2])
    x3 = np.array([[x] for x in x3])
    # x=np.hstack((x1,x2,x3))
    x = np.hstack((x11,x2,x3))
    print("result")
    # y_shuffle = zeroNormalize(newdata["volume"])
    y_shuffle = newdata["volume"]
    y = data["volume"]
    # plt.plot(data["season"]+data["trend"],color='g')
    # plt.plot(data["neighbour"],color = 'y')
    # plt.plot(y, color='red')
    # print(my_custom_loss_func(y,data["neighbour"]))
    
    # plt.show()
    

    clf = LinearSVR(C=1, epsilon=0.1)
    # clf = RandomForestRegressor(n_estimators = 100, n_jobs = -1,random_state =50,
                                    # max_features = "auto", min_samples_leaf = 1)
    clf.fit(x, y_shuffle)   
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y_shuffle,cv=10,scoring=score)
    print(scores)
    print(np.mean(scores))
    

def create_main():
    trend_cols = selectfeature(trend_path, 0)
    residual_cols = 1
    return trend_cols,residual_cols

def selectfeature(source_path, isResidual):
    x,y,cols= get_source_data(source_path, isResidual) 
    clf = RandomForestRegressor(n_estimators=10, random_state = 0)
    clf.fit(x,y)
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    for f in range(10):
        print("%2d) %-*s %f" %(f+1, 30,cols[indices[f]],importance[indices[f]]))
    if isResidual:
        x_selected = clf.transform(x,threshold = importance[indices[4]])
    else:
        x_selected = clf.transform(x,threshold = 0.01)
    featurenums = x_selected.shape[1]
    feature_dic = {}
    for f in range(featurenums):
        feature_dic[cols[indices[f]]] = 1
    # print(feature_dic)
    selected_cols=[]
    for c in cols:
        if c in feature_dic:
            selected_cols.append(c)
    print(selected_cols)
    
    # print(x_selected.shape)
    # print(importance[indices[3]])
    creat_model1(x_selected, y, isResidual)
    return selected_cols
    
def creat_model1(x,y, isResidual):
    # clf = SVR(C=1.0, epsilon=0.1)
    sample_leaf_options = [1,5,10,50,100,200,500]

    # for leaf_size in sample_leaf_options :
    if isResidual:
        # clf = SVR(C=10.0, epsilon=0.1)
        # score = make_scorer(my_custom_loss_func, greater_is_better=False)
        # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
        # tuned_parameters = {'n_estimators': [100,500, 700, 1000], 'max_depth': [None, 1, 2, 3], 'min_samples_leaf': sample_leaf_options}
        # gs = GridSearchCV(estimator = RandomForestRegressor(), param_grid = tuned_parameters, cv = 10)
        # gs = gs.fit(x,y)
        # print(gs.best_score_)
        # print(gs.best_params_)
        
        # print(scores)
        clf = RandomForestRegressor(n_estimators = 100, n_jobs = -1,random_state =50,
                                    max_features = "auto", min_samples_leaf = 1)
        # score = make_scorer(my_custom_loss_func, greater_is_better=False)
        # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
        # print(scores)
        # clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
        # score = make_scorer(my_custom_loss_func, greater_is_better=False)
        # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
        # print(scores)
    else:
        clf = RandomForestRegressor(n_estimators = 200, oob_score = True, n_jobs = -1,random_state =50,
                                    max_features = "auto", min_samples_leaf = 10)

    clf.fit(x,y)
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    mean_score = np.mean(scores)
    print(mean_score)
    if isResidual:
        joblib.dump(clf, residual_model_path)
    else:
        joblib.dump(clf, trend_model_path)
    
    # clf = RandomForestRegressor(n_estimators=10,oob_score = TRUE,n_jobs = -1,random_state =1)
    # clf.fit(x,y)
    # clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    
    # clf = AdaBoostRegressor(n_estimators=100, base_estimator=rg,learning_rate=1)
    # clf.fit(x, y)   
    # score = make_scorer(my_custom_loss_func, greater_is_better=False)
    # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    # print(scores)
    # joblib.dump(clf, 'F:/kdd/scripts/svrvolume.pkl')
    # clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    # clf.fit(x,y)
    # score = make_scorer(my_custom_loss_func, greater_is_better=False)
    # scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    # print(scores)
    # joblib.dump(clf, 'F:/kdd/scripts/gbtvolume.pkl')
    
if __name__ == "__main__":
    # creat_model('F:/kdd/dataSets/training/voulumn_aggregate_data.csv',"",1)
    # selectfeature(residual_path, 1)
    selectfeature(trend_path, 0)
    
    
   
