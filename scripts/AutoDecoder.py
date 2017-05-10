import tensorflow as tf
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

input_nodes = 72 
hidden_size = 36
output_nodes = 72 
train_path = "F:/kdd/dataSets/training/training_20min_avg_volume_update.csv"

def get_train_data(id,direction):
    data = pd.read_csv(train_path,encoding='utf-8')
    partial_data = data['volume'][(data['tollgate_id']==id)&(data['direction'] == direction)]
    return partial_data

def computecost(w,b,x,w1,b1):
    p = 0.01
    # beta = 0.001
    beta = 0#0.0001#sparsity
    lamda = 0#0.0001#weight decay
    
    hidden_output = tf.sigmoid(tf.matmul(x,w) + b)
    pj = tf.reduce_mean(hidden_output, 0)
    sparse_cost = tf.reduce_sum(p*tf.log(p/pj)+(1-p)*tf.log((1-p)/(1-pj)))
    output = tf.sigmoid(tf.matmul(hidden_output,w1)+b1)
    regular = lamda*(tf.reduce_sum(w*w)+tf.reduce_sum(w1*w1))/2
    cross_entropy = tf.reduce_mean(tf.pow(output - x, 2))/2 +sparse_cost*beta + regular #+ regular+sparse_cost*beta 
    return cross_entropy, hidden_output, output

def xvaier_init(input_size, output_size):
    low = -np.sqrt(6.0/(input_nodes+output_nodes))
    high = -low
    return tf.random_uniform((input_size, output_size), low, high, dtype = tf.float32)

    
def main():
    w = tf.Variable(xvaier_init(input_nodes, hidden_size))
    b = tf.Variable(tf.truncated_normal([hidden_size],0.1))   
    x = tf.placeholder(tf.float32, shape = [None, input_nodes])
    w1 = tf.Variable(tf.truncated_normal([hidden_size,input_nodes], -0.1, 0.1))
    b1 = tf.Variable(tf.truncated_normal([output_nodes],0.1))

    cost, hidden_output, output = computecost(w,b,x,w1,b1)

    train_step = tf.train.AdamOptimizer(1).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
   
    partial_data = get_train_data(1,0)
    train_x = []
    for i in range(len(partial_data)-72):
        train_x.append(partial_data[i:i+72])
    train_x = np.array(train_x)
    print(train_x.shape)

    for i in range(10000):
        _,hidden_output_, output_,cost_,w_= sess.run([train_step, hidden_output, output,cost,w], feed_dict = {x : train_x})
        if i%1000 == 0:
            print(hidden_output_)
            print(output_)
            print(cost_)

    # np.save("weights1.npy", w_)
    # show_image(w_)
        
            
if __name__ == '__main__':
    main()
            
    
    
    
    