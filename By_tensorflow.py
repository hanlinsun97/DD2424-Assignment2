import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def LoadBatch(dataset_name):
    dataset = sio.loadmat(dataset_name)
    data = np.float32(np.array(dataset['data'])) / np.max(np.float32(np.array(dataset['data'])))
    labels_1 = np.array(dataset['labels'])
    label_no_onehot = []
    for j in range(np.size(labels_1)):
        label_no_onehot.append(labels_1[j][0])
    label_no_onehot = np.array(label_no_onehot)
    data_length = np.size(labels_1)
    label_max = 10
    label = np.zeros([np.size(labels_1),label_max])
    for i in range(np.size(labels_1)):
        label[i][labels_1[i]] = 1
    data = np.transpose(data) # 3072 * 10000
    label = np.transpose(label) # 10000 * 10
    return data, label, data_length, label_no_onehot

def initialization(m):
    W1 = tf.Variable(tf.random_normal([m, 3072])/1000)
    W2 = tf.Variable(tf.random_normal([10,m])/1000)
    b1 = tf.Variable(tf.zeros([m,1]))
    b2 = tf.Variable(tf.zeros([10,1]))
    return W1, W2, b1, b2

def ComputeAccuracy(P, input_label_no_onehot, batch_size):
    Q = P.argmax(axis=0)  # Predict label
    #print(Q)
    Y = input_label_no_onehot
    diff = Q - Y
    acc = np.sum(diff == 0)/batch_size
    return acc


learning_rate = 0.05
batch_size = 100
training_epoch = 50
m = 150
lam = 0.006

[W1, W2, b1, b2] = initialization(m)
X = tf.placeholder("float32",[3072,None])
Y = tf.placeholder("float32",[10, None])
s1 = tf.matmul(W1, X) + b1
h = tf.nn.relu(s1)
s = tf.matmul(W2, h) + b2
p = tf.nn.softmax(s)
L = -tf.reduce_sum(Y*tf.log(p))/batch_size
J = L + lam * (tf.reduce_sum(W1**2)+tf.reduce_sum(W2**2))

optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(J)

[data_1, label_1, data_length_1, label_no_onehot_1] = LoadBatch("data_batch_1.mat")
[data_2, label_2, data_length_2, label_no_onehot_2] = LoadBatch("test_batch.mat")

data_1_mean = np.mean(data_1,1)
data_1_mean = np.reshape(data_1_mean,[3072,1])
data_1 = data_1 - data_1_mean
data_2 = data_2 - data_1_mean

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(data_length_1/batch_size)
    for epoch in range(training_epoch):
        print(epoch)
        for i in range(total_batch):
            batch_x = data_1[:,i*batch_size:(i+1)*batch_size]
            batch_y = label_1[:,i*batch_size:(i+1)*batch_size]
            _, loss,P = sess.run([optimizer, L, p],feed_dict={X:batch_x, Y:batch_y})
        
        batch_xs = data_1
        batch_ys = label_1
        loss,P = sess.run([L, p],feed_dict={X:batch_xs, Y:batch_ys})
        acc = ComputeAccuracy(P, label_no_onehot_1, data_length_1)
        print(loss/100)
        print("Acc on training data",acc)
         
        batch_xss = data_2
        batch_yss = label_2
        loss,P = sess.run([L, p],feed_dict={X:batch_xss, Y:batch_yss})
        acc = ComputeAccuracy(P, label_no_onehot_2, data_length_2)
        print(loss/100)
        print("Acc on validation data",acc)