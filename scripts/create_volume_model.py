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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
from sklearn.ensemble import GradientBoostingRegressor

residual_path = "F:/kdd/dataSets/training/residual_norm.csv"
trend_path = "F:/kdd/dataSets/training/trend_norm.csv"

residual_model_path = "F:/kdd/dataSets/training/residual.pkl"
trend_model_path = "F:/kdd/dataSets/training/trend.pkl" 

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'epsilon':[0.1,0.2,0.3],
                     'C': [1, 10, 100]}]

def get_source_data(norm_data_path, isResidual):
    data = pd.read_csv(norm_data_path,encoding='utf-8')
    # data = data.sample(frac=1)
    
    tempcols = []
    for i in range(7):
        tempcols.append(str(i))
    tempcols = np.array(tempcols)
    restcols = np.array(['holiday', 'norm_time', 'pressure','sea_pressure','wind_direction','temperature','rel_humidity','precipitation'])
    cols = np.hstack((tempcols, restcols))
    x = data[cols]
    
    # if isResidual:
        # x = data[['norm_time','holiday','pressure', 'sea_pressure',
        # 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation']]
    # else:
        # x = data[['norm_time','holiday','pressure', 'sea_pressure',
        # 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation']]
    y = data["volume"]

    # plt.plot(y)
    # plt.show()
    return x,y,cols
    
def my_custom_loss_func(ground_truth, predictions):
    return np.mean(np.abs((ground_truth-predictions)/(ground_truth)))
    
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
    
def create_main():
    creat_model(residual_path, residual_model_path, 1)
    creat_model(trend_path, trend_model_path, 0)
    
    
def selectfeature(source_path, isResidual):
    x,y,cols= get_source_data(source_path, isResidual) 
    clf = RandomForestRegressor(n_estimators=10, random_state = 0)
    clf.fit(x,y)
    importance = clf.feature_importances_
    # print(importance)
    indices = np.argsort(importance)[::-1]
    # print(indices)
    for f in range(10):
        print("%2d) %-*s %f" %(f+1, 30,cols[indices[f]],importance[indices[f]]))
    if isResidual:
        x_selected = clf.transform(x,threshold = 0.04)
    else:
        x_selected = clf.transform(x,threshold = 0.01)
    print(x_selected.shape)
    if isResidual:
        plt.plot(y)
        plt.show()
    # creat_model(x_selected, y)
    
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
    joblib.dump(clf, 'F:/kdd/scripts/rfvolume.pkl')
    # clf = RandomForestRegressor(n_estimators=10,oob_score = TRUE,n_jobs = -1,random_state =1)
    # clf.fit(x,y)
    # clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    clf = SVR(C=1.0, epsilon=0.1)
    # clf = AdaBoostRegressor(n_estimators=100, base_estimator=rg,learning_rate=1)
    clf.fit(x, y)   
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    joblib.dump(clf, 'F:/kdd/scripts/svrvolume.pkl')
    clf =  GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1,max_features = None, max_depth = 3, random_state = 1)
    clf.fit(x,y)
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
    print(scores)
    joblib.dump(clf, 'F:/kdd/scripts/gbtvolume.pkl')
    
if __name__ == "__main__":
    # creat_model('F:/kdd/dataSets/training/voulumn_aggregate_data.csv',"",0)
    selectfeature(residual_path, 1)
    selectfeature(trend_path, 0)
    
    
   
