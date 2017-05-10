import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import numpy as np

import tensorflow as tf

num_steps = 6
hidden_size = 10
inputdim = 1
x = tf.placeholder(tf.float32, [None, num_steps, inputdim])
y = tf.placeholder(tf.float32, [None, num_steps])
w = tf.Variable(tf.random_normal([hidden_size, 1]))
b = tf.Variable(tf.random_normal([1]))
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
W_repeated = tf.tile(tf.expand_dims(w, 0), [tf.shape(x)[0], 1, 1])
out = tf.matmul(outputs, W_repeated) + b#outputs none*num_steps*hidden_size
out = tf.squeeze(out)#out none*num_steps*1

predict_data = pd.read_csv(volume_test_path, encoding='utf-8') 
real_data = pd.read_csv(real_volume_path, encoding='utf-8')

def gettruevolume():
    time_windows = real_data['time_window'][(real_data['tollgate_id']==id)&(real_data['direction'] == direction)]
    # print(time_windows)
    volumes = real_data['volume'][(real_data['tollgate_id']==id)&(real_data['direction'] == direction)]
    l = len(time_windows)
    time_windows = np.array(time_windows)
    volumes = np.array(volumes)
    
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
    return res_dic,time_windows     

def predic():
    saver = tf.train.Saver(tf.global_variables())#之前有variable操作 
    with tf.Session() as session:
        saver.restore(session, 'F:/kdd/scripts/ass.model')
        # tf.get_variable_scope().reuse_variables()
        # session.run(tf.global_variables_initializer())
        prev_seq = train_x[-1]
        predict = []
        for i in range(72):
            new_out,wb = session.run([out,w], feed_dict = {x:[prev_seq]})
            predict.append(new_out[-1])
            prev_seq = np.vstack((prev_seq[1:], new_out[-1]))
        # plt.figure()
        # plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
        # plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
        # plt.show()

