import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime,timedelta

dates = np.array(["2016/10/18","2016/10/19","2016/10/20","2016/10/21","2016/10/22","2016/10/23","2016/10/24"])
intervals = np.array([24,25,26,27,28,29,51,52,53,54,55,56]) 
intervals = np.array([18,19,20,21,22,23,45,46,47,48,49,50]) 

num_steps = 6
inputdim = 1
keep_prob = 1.0
hidden_size = 100
num_layers = 3

from getPath import *
pardir = getparentdir()

real_volume_path = pardir+"/dataSets/testing_phase1/test1_20min_avg_volume_update.csv"
real_data = pd.read_csv(real_volume_path, encoding='utf-8')

residual_path = pardir+"/dataSets/training/residual.csv"
marks = pd.read_csv(residual_path,encoding='utf-8')
data = marks['volume']

normalized_data = (data - np.mean(data)) / np.std(data)
mean_data = np.mean(data)
std_data = np.std(data)
# normalized_data = data

datalen = len(data)
train_x = []
train_y = []
for i in range(datalen-num_steps-1):
    train_x.append(np.expand_dims(normalized_data[i : i + num_steps], axis=1).tolist())
    train_y.append(normalized_data[i+1:i+num_steps+1].tolist())
    
x = tf.placeholder(tf.float32, [None, num_steps, inputdim])#??? train和predict不能一起
y = tf.placeholder(tf.float32, [None, num_steps])
w = tf.Variable(tf.random_normal([hidden_size, 1]))
b = tf.Variable(tf.random_normal([1]))
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0)
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = keep_prob)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell], num_layers)

# initial_state = cell.zero_state(1, tf.float32)
# state =initial_state

outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
W_repeated = tf.tile(tf.expand_dims(w, 0), [tf.shape(x)[0], 1, 1])
out = tf.matmul(outputs, W_repeated) + b#outputs none*num_steps*hidden_size
out = tf.squeeze(out)#out none*num_steps*1

def gettruevolume(id, direction):
    time_windows = real_data['time_window'][(real_data['tollgate_id']==id)&(real_data['direction'] == direction)]
    volumes = real_data['volume'][(real_data['tollgate_id']==id)&(real_data['direction'] == direction)]
    l = len(time_windows)
    time_windows = np.array(time_windows)
    volumes = np.array(volumes)
    # print(volumes)
    
    mean_true_data = np.mean(volumes)
    std_true_data = np.std(volumes)
    volumes = (volumes - mean_true_data) / std_true_data
    # print(volumes)
    res_dic = {}
    for i in range(l):
        time = time_windows[i]
        trace_time = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
        date = str(trace_time.year)+'/'+str(trace_time.month)+'/'+str(trace_time.day)
        num = (trace_time.hour*60+trace_time.minute)/20
        if num==0:
            num = 72
        if not date in res_dic:
            res_dic[date]={}
        res_dic[date][num] = volumes[i]
    return volumes, mean_true_data,std_true_data,res_dic

def get_source_data(norm_data_path):
    data = pd.read_csv(norm_data_path,encoding='utf-8')
    y = data["volume"]
    return y
  
def train_rnn():   
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # tf.get_variable_scope().reuse_variables()
        loss = tf.reduce_mean(tf.square(out - y))
        train_op = tf.train.AdamOptimizer().minimize(loss)
        session.run(tf.global_variables_initializer())
        for step in range(10000):
            _,los,wa= session.run([train_op, loss, w], feed_dict = {x: train_x,y: train_y})
            if step % 100 == 0:
                print("step"+ str(step)+ ":"+str(los))
        saver.save(session,'F:/kdd/scripts/ass.model')
        print(wa)
        prev_seq = train_x[-1]
        # print(prev_seq)
        predict = []
        for i in range(72*7):
            new_out,wb = session.run([out,w], feed_dict = {x:[prev_seq]})
            predict.append(new_out[-1])
            prev_seq = np.vstack((prev_seq[1:], new_out[-1]))
        # print(wb)
        plt.figure()
        plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
        plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
        predict = np.array(predict)
        update_predict = predict*std_data+mean_data
        writeResultTofile(update_predict)
        plt.show()
        
def writeResultTofile(values):
    file_path = pardir+"/dataSets/training/lstmResult.csv"
    dates = np.array(["2016/10/18","2016/10/19","2016/10/20","2016/10/21","2016/10/22","2016/10/23","2016/10/24"]) 
    fw = open(file_path, 'w')
    l = len(values)
    fw.writelines(','.join(['"date"', '"interval"', '"volume"'])+'\n')
    for i in range(l):
        interval = i%72
        if i%72==0:
            date = dates[i/72]
            interval = 72
        outline = ','.join(['"' + date + '"', '"' + str(interval) + '"', '"' + str(values[i]) + '"',]) + '\n'
        fw.writelines(outline)
    fw.close()
    
        
def predic():
    saver = tf.train.Saver(tf.global_variables())#之前有variable操作
    with tf.Session() as session:
        saver.restore(session, 'F:/kdd/scripts/ass.model')
        # tf.get_variable_scope().reuse_variables()
        # session.run(tf.global_variables_initializer())
        prev_seq = train_x[-1]
        # print(prev_seq)
        predict = []
        for i in range(72*7):
            new_out,wb = session.run([out,w], feed_dict = {x:[prev_seq]})
            predict.append(new_out[-1])
            prev_seq = np.vstack((prev_seq[1:], new_out[-1]))
        # print(wb)
        # plt.figure()
        # plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
        # plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
        # plt.show()
    return predict
        
def predictfromData(id, direction):#有问题
    dic = {}
    volumes, mean_true_data,std_true_data, res_dic = gettruevolume(id, direction)
    datelen = len(dates)
    new_train_x = []
    
    for i in range(datelen):
        new_train_x.append(np.expand_dims(volumes[i*12 : i*12+num_steps], axis=1).tolist())
        new_train_x.append(np.expand_dims(volumes[i*12+num_steps : i*12+2*num_steps], axis=1).tolist())
        
    saver = tf.train.Saver(tf.global_variables())#之前有variable操作
    with tf.Session() as session:
        saver.restore(session, 'F:/kdd/scripts/ass.model')
        # tf.get_variable_scope().reuse_variables()
        # session.run(tf.global_variables_initializer())

        predict = []
        for date in dates:
            k = 0
            for i in range(len(intervals)):
                if i%6==0:
                    temp = int(i/6)
                    prev_seq = new_train_x[k+temp]
                new_out,wb = session.run([out,w], feed_dict = {x:[prev_seq]})
                predict.append(new_out[-1])
                prev_seq = np.vstack((prev_seq[1:], new_out[-1]))
            k+=12
        index = 0
        for date in dates:
            for interval in intervals:
                if not date in dic:
                    dic[date] = {}
                dic[date][interval] = predict[index]*std_true_data+mean_true_data
                index += 1
    return dic
    
def predictTest(id, direction):
    dic = {}
    volumes, mean_true_data,std_true_data,res_dic  = gettruevolume(id, direction)
    datelen = len(dates)
    new_train_x = []
    
    for i in range(datelen):
        new_train_x.append(np.expand_dims(volumes[i*12 : i*12+num_steps], axis=1).tolist())
        new_train_x.append(np.expand_dims(volumes[i*12+num_steps : i*12+2*num_steps], axis=1).tolist())
        
        
    saver = tf.train.Saver(tf.global_variables())#之前有variable操作
    with tf.Session() as session: 
        saver.restore(session, 'F:/kdd/scripts/ass.model')
        # tf.get_variable_scope().reuse_variables()
        # session.run(tf.global_variables_initializer())
        l = len(new_train_x)
        print(l)
        predict = []
        final_res = []
        for i in range(l):
            if i%2==0:
                prev_seq = new_train_x[i]
                for j in range(27):
                    new_out= session.run(out, feed_dict = {x:[prev_seq]})
                    predict.append(new_out[-1])
                    prev_seq = np.vstack((prev_seq[1:], new_out[-1]))
                for k in range(-6,0):
                    final_res.append(predict[k])
            
        final_res = np.array(final_res)
        # final_res = final_res*std_true_data + mean_true_data       
    return final_res
        
def comparelstm(id,direction):
    volumes, mean_true_data,std_true_data,res_dic = gettruevolume(id, direction)
    predictlong = predic()
    predictshort = predictTest(id, direction)
    res = []
    for i in range(72*7):
        if i%72 == 45:
            for j in range(i,i+6):
                res.append(predictlong[j])
    res = np.array(res)
    true_value = []
    for date in dates:
        for interval in intervals:
            if not date in res_dic:
                continue
            if not interval in res_dic[date]:
                continue
            if(interval == 45):
                for i in range(interval, interval+6):
                    true_value.append(res_dic[date][i])
                break
                    
    true_value = np.array(true_value)
    true_value = true_value*std_true_data+mean_true_data
    res = res*std_data+mean_data
    predictshort = predictshort*std_true_data+mean_true_data
    
    print(my_custom_loss_func(true_value, res))
    print(my_custom_loss_func(true_value, predictshort))
    
    # print(true_value)
    plt.plot(res, color = 'orange')
    plt.plot(predictshort, color = 'blue')
    plt.plot(true_value, color='red')
    plt.show()
def my_custom_loss_func(ground_truth, predictions):
    return np.mean(np.abs(ground_truth-predictions)/ground_truth)            
    
            
def compare():
    ids = [1,2,3]
    directions = [0,1]
    for id in ids:
        for direction in directions:
            if id==2 and direction==1:
                continue
            comparelstm(id,direction)
# train_rnn()
# predic()        
        

# compare()        
    

