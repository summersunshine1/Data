import pandas as pd
from datetime import datetime,timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import arma_order_select_ic
from sklearn.metrics import mean_squared_error

data_path = "F:/kdd/dataSets/training/training_20min_avg_volume_update.csv"
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
data = pd.read_csv(data_path, parse_dates=['time_window'], index_col='time_window',date_parser=dateparse)


test_data_path = "F:/kdd/dataSets/testing_phase1/test1_20min_avg_volume_update.csv"
test_data = pd.read_csv(test_data_path, parse_dates=['time_window'], index_col='time_window',date_parser=dateparse)

residual_path = "F:/kdd/dataSets/training/residual.csv"
seasonal_path = "F:/kdd/dataSets/training/seasonal.csv"
trend_path = "F:/kdd/dataSets/training/trend.csv"

def writevolume(volume, path):
    fw = open(path, 'w')
    fw.writelines(','.join(['"time_window"', '"residual"']) + '\n')
    l = len(residual)
    indexs = residual.index
    for i in range(l):
        out_line = ','.join(['"' + str(indexs[i]) + '"', '"' + str(residual.loc[indexs[i]][0]) + '"']) + '\n'
        fw.writelines(out_line)
    fw.close()
    
    
    
def getData(id, direction):
    partial_data = data[['volume']][(data['tollgate_id']==id)&(data['direction'] == direction)]
    return partial_data
    
def getTestData(id,direction):
    partial_data = test_data[['volume']][(test_data['tollgate_id']==id)&(test_data['direction'] == direction)]
    # print(partial_data)
    # partial_data = data[['volume']]
    return partial_data
    
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(center=False, window=72).mean()
    rolstd = timeseries.rolling(center=False, window=72).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    
    
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries)#, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    plt.show(block=True)
    
def decompose(ts_log):
    
    decomposition = seasonal_decompose(ts_log,model = "multiplicative",freq=72)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    seasonal.dropna(inplace=True)
    trend.dropna(inplace=True)
    residual.dropna(inplace=True)
    # plt.subplot(411) 
    # plt.plot(ts_log, label='Original')
    # plt.legend(loc='best')
    # plt.subplot(412)
    # plt.plot(trend, label='Trend')
    # plt.legend(loc='best')
    # plt.subplot(413)
    # plt.plot(seasonal,label='Seasonality')
    # plt.legend(loc='best')
    # plt.subplot(414)
    # plt.plot(residual, label='Residuals')
    # plt.legend(loc='best')
    # plt.tight_layout()
    ts_log_decompose = residual
    plt.plot(seasonal*residual*trend, color = 'blue')
    plt.plot(ts_log, color = 'red')
    

    plt.show()
    return ts_log_decompose,trend,seasonal
 
    
def acfplot(ts_log_diff):
    lag_acf = acf(ts_log_diff, nlags=25)
    lag_pacf = pacf(ts_log_diff, nlags=25, method='ols')

    #Plot ACF:    
    # plt.subplot(121)    
    # plt.plot(lag_acf)
    
    # plt.axhline(y=0,linestyle='--',color='gray')
    # plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    # plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    # plt.title('Autocorrelation Function')

    #Plot PACF:
    # plt.subplot(122)
    # plt.plot(lag_pacf)
    # fig = sm.graphics.tsa.plot_pacf(dta,lags=40,ax=ax2)
    # plt.axhline(y=0,linestyle='--',color='gray')
    # plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    # plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    # plt.title('Partial Autocorrelation Function')
    # plt.tight_layout()
    fig = plt.figure(figsize=(12,8))
    ax1=fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts_log_diff,lags=20,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts_log_diff,lags=20,ax=ax2)
    plt.show()
 
def my_custom_loss_func(ground_truth, predictions):
    return np.mean(np.abs(ground_truth-predictions)/ground_truth)
    
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    # print(train)
    history = [x for x in train]
    # print(history)
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error
    
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset['volume']
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                # try:
                mse = evaluate_arima_model(dataset, order)
                if mse < best_score:
                    best_score, best_cfg = mse, order
                print('ARIMA%s MSE=%.3f' % (order,mse))
                # except as err:
                	# print(err)
                # continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    
ts = getData(1,0)
plt.plot(ts,color = 'green')
plt.plot
plt.show()
ts = getData(1,1)
plt.plot(ts,color = 'green')
plt.plot
plt.show()
ts = getData(2,0)
plt.plot(ts,color = 'green')
plt.plot
plt.show()
ts = getData(3,0)
plt.plot(ts,color = 'green')
plt.plot
plt.show()
ts = getData(3,1)
plt.plot(ts,color = 'green')
plt.plot
plt.show()
# test_stationarity(ts)

# ts_log = np.log(ts)
# test_stationarity(ts_log)

# ts_log_diff = ts_log - ts_log.shift(72)
# ts_log_diff.dropna(inplace=True)
# test_stationarity(ts_log_diff)
# ts_log_decompose,trend,seasonal = decompose(ts)
# ts_log_decompose = ts_log_decompose.diff(1)
# plt.plot(trend+ts_log_decompose)
# plt.plot(ts_log_decompose,color = 'red')
# plt.plot
# plt.plot(trend,color = 'yellow')
# plt.plot(ts_log_decompose, color = 'green')
# plt.plot(ts_log_decompose)
# plt.show()
# ts_log_decompose.dropna(inplace=True)
# p_values = [0, 1, 2, 4, 6, 8, 10]
# d_values = range(0, 3)
# q_values = range(0, 3)
# evaluate_models(ts_log_decompose, p_values, d_values, q_values)


# acfplot(ts_log_decompose)
# test_stationarity(ts_log_decompose)
# ts = getData(1,1)
# ts_log_decompose,trend,seasonal = decompose(ts)
# plt.plot(trend)
# plt.plot(ts_log_decompose,color = 'red')
# plt.show()
# ts = getData(2,0)
# ts_log_decompose,trend,seasonal = decompose(ts)
# print(trend)
# print(seasonal)
# print(ts_log_decompose)
# plt.plot(trend)
# plt.plot(seasonal)
# plt.plot(ts_log_decompose,color = 'red')
# plt.show()
# ts = getData(3,0)
# ts_log_decompose,trend,seasonal = decompose(ts)
# plt.plot(trend)
# plt.plot(ts_log_decompose,color = 'red')
# plt.show()
# ts = getData(3,1)
# ts_log_decompose,trend,seasonal = decompose(ts)
# plt.plot(trend)
# plt.plot(ts_log_decompose,color = 'red')
# plt.show()
# print(seasonal)
# writevolume(ts_log_decompose, "")
# test_stationarity(ts_log_decompose)
# 

# temp = ts_log_decompose[:-36]
# print(temp)
# print(temp)
# temp.dropna(inplace=True)
# temp = np.log(temp)
# temp.dropna(inplace=True)
# test_stationarity(temp)
# print(temp)
# print(temp)
# acfplot(temp)
# res = arma_order_select_ic(temp,10,10,ic=['aic','bic'],trend='nc')
# print(res.bic_min_order)
# print(temp)


# test_stationarity(temp)
# plt.plot(temp-temp.shift(72))
# plt.show()
# index = temp.index
# pd.DatetimeIndex.append(index,pd.DatetimeIndex(['2016-10-18 00:00:00'])) 
# print('model')
# model = ARIMA(temp, order=(0,0,2))  
# results_ARIMA = model.fit(disp=-1)  
# print("results:"+str(results_ARIMA.predict()))
# print('model')
# print(temp)
# recovered_value = results_ARIMA.fittedvalues
# df = pd.DataFrame(recovered_value)

# predictions_ARIMA_diff = pd.Series(f, copy=True)
# plt.plot(temp, color = 'red')
# plt.plot(recovered_value)
# plt.show()

time = []
# start_time = '2016-10-17 12:00:00'
start_time = '2016-10-17 00:00:00'
trace_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
start_time_window = datetime(trace_time.year, trace_time.month, trace_time.day,trace_time.hour, trace_time.minute, 0)
time.append(start_time_window)
end_time_window = start_time_window
for i in range(35):
    end_time_window = end_time_window + timedelta(minutes = 20)
    time.append(end_time_window)
time_index = pd.DatetimeIndex(time)
# print(time_index)
 
# rest = ts_log_decompose[-36:]


# restaverage = pd.ewma(rest, halflife=12)
# temprest = ts_log_decompose[-42:]
# moving_average = pd.rolling_mean(temprest,6)
# f,_,_ = results_ARIMA.forecast(36)
f = []
tempvolume = temp['volume']
history = [t for t in tempvolume]
for t in range(36):
    model = ARIMA(history, order=(0,0,5))
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    f.append(yhat)
    history.append(yhat)


f = pd.DataFrame(f)
f.index = time_index
f.columns = ['volume']
# print(f)
# s = seasonal[-35:]['volume']

# s = pd.DataFrame(s)
# s.index = time_index
# f= f+s
# f = f.shift(-1)
# plt.plot(f)
# plt.plot(restaverage, color = 'yellow')
# plt.plot(rest,color='red')
# plt.plot(moving_average, color = 'blue')
# plt.plot((restaverage+f+moving_average)/3, color = 'black')
# print(np.mean(np.abs((rest['volume']-f['volume'])/rest['volume'])))
# print(np.mean(np.abs((rest['volume']-restaverage['volume'])/rest['volume'])))
# print(np.mean(np.abs((rest['volume']-moving_average['volume'])/rest['volume'])))
# print(np.mean(np.abs((rest['volume']-(restaverage['volume']+f['volume']+moving_average['volume'])/3)/rest['volume'])))
# plt.show()
# f = f.shift(-1)
# print(f)


# ground_truth = []
# predicted_value = []
temp_trend = ts_log_decompose[-36:]
# temp_trend = temp_trend.diff(1)
# temp_trend = temp_trend[1:]
# plt.plot(f)
# plt.plot(temp_trend,color='red')
# plt.show()
# print(my_custom_loss_func(temp_trend,f))

# for i in range(len(time)-1):
    # ground_truth.append(np.exp(temp_trend.loc[time[i]][0]))
    # predicted_value.append(np.exp(f.loc[time[i]][0]))
    


# test_data = getTestData(1,0)
# start_time = '2016-10-18 06:00:00'
# trace_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
# start_time_window = datetime(trace_time.year, trace_time.month, trace_time.day,trace_time.hour, trace_time.minute, 0)
# time.append(start_time_window)
# end_time_window = start_time_window
# ground_truth = []
# predicted_value = []
# for i in range(5):
    # end_time_window = end_time_window + timedelta(minutes = 20)
    # ground_truth.append((test_data.loc[end_time_window][0]))
    # predicted_value.append(np.exp(f.loc[end_time_window][0]))

# ground_truth = np.array(ground_truth)
# predicted_value = np.array(predicted_value)
# error = np.mean(np.abs(ground_truth-predicted_value)/ground_truth)
# print(error)



# plt.show()

# plt.plot(test_data, color='red')
# plt.plot(f)
# plt.show()


                                    
# index = pd.DatetimeIndex(['2016-10-17 12:00:00','2016-10-17 12:20:00'])
# df.columns = ['volume']
# plt.plot(df+seasonal, 'red')
# plt.show()
# recovered_value.dropna(inplace=True)



# df = seasonal+df
# df.columns = ['volume']
# plt.plot(trend)
# plt.plot(np.exp(ts_log))
# plt.plot(np.exp(seasonal+df), color = 'red')
# df = df+seasonal
# df = df.shift(-1)
# print(ts_log)
# print(seasonal+df)

# error = np.mean(np.abs(ts_log-seasonal-df)/ts_log)
# print(error)
# error = np.mean(np.abs(np.exp(ts_log)-np.exp(df))/np.exp(ts_log))
# print(error)
# plt.show()
# se_index = seasonal.index

