import sys
sys.path.append('E:/kdd/Data/scripts/common')
from commonLib import *
import pandas as pd

weather_train_path = "E:/kdd/Data/dataSets/training/weather (table 7)_training_update.csv"
weather_test_path = "E:/kdd/Data/dataSets/testing_phase1/weather (table 7)_test1.csv"

def get_train_test(colume, train_data, test_data):
    train_data = [d for d in train_data[colume]]
    test_data = [d for d in test_data[colume]]
    data = train_data+test_data
    return data

def get_Discrete_Weather(colume = 'pressure', bins = 10):
    train_data = pd.read_csv(weather_train_path,encoding='utf-8')
    test_data = pd.read_csv(weather_test_path,encoding='utf-8')
    
    dates = get_train_test('date', train_data, test_data)
    hours = get_train_test('hour', train_data, test_data)

    train = [d for d in train_data[colume]]
    test = [d for d in test_data[colume]]
    totaldata = train + test
    arr = [a for a in range(bins)]
    out = pd.cut(totaldata, bins, labels = arr)
    out = [[o] for o in out]
    res = encoder(out)
    
    traindic = {}
    testdic = {}
    
    for i in range(len(train)):
        if not dates[i] in traindic:
            traindic[dates[i]]={}
        traindic[dates[i]][hours[i]] = res[i]
    for i in range(len(train),len(res)):
        if not dates[i] in testdic:
            testdic[dates[i]]={}
        testdic[dates[i]][hours[i]] = res[i]
    return traindic,testdic
    
def get_Discrete_Normtime(bins = 72):
    arr = [[a] for a in range(bins)]
    res = encoder(arr)
    return res
    
if __name__ == "__main__":
    # getDiscreteWeather()
    print(get_Discrete_Normtime())

    
    
    
