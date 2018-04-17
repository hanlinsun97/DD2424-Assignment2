import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# define some functions

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
    # W1 = np.random.normal(0,0.01,[m,3072])
    # W2 = np.random.normal(0,0.01,[10,m])

    W1 = np.random.randn(m,3072)/np.sqrt(3072/2)
    W2 = np.random.randn(10,m)/np.sqrt(m/2)
    b1 = np.random.normal(0,0.01,[m,1])
    b2 = np.random.normal(0,0.01,[10,1])
    return W1, W2, b1, b2

def ComputeCost(label, lam, batch_size, P, W1, W2):
    Y = label
    loss = -(1.0 / batch_size) * np.sum(Y * np.log(P))
    J = loss + lam * (np.sum(np.power(W1, 2)) + np.sum(np.power(W2, 2)))
    #J = loss+lam*np.sum(W**2)
    return J, loss

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def Compute_S(W, b, input_data):
    s = np.dot(W, input_data) + b
    return s

def Compute_h(s):       #ReLU
    h = np.maximum(0,s)
    return h


def EvaluationClassifier(s):
    P = softmax(s)
    return P


def ComputeGradients(W1, W2, b1, b2, P, input_data, input_label, lam, batch_size,m,h):

    # input_data with 3072 * Batch_size
    # input_label with 10 * Batch_size
    # g = np.mean(P - input_label,1)

    g = P - input_label
    grad_b2= np.mean(g,1)
    grad_W2 = np.dot(g,h.T)/batch_size + 2 * lam * W2

    g = np.dot(W2.T, g)
    h[h > 0] = 1
    g = g * h
    grad_b1 = np.mean(g,1)
    grad_W1 = np.dot(g,input_data.T)/batch_size + 2 * lam * W1
    grad_b1 = np.reshape(grad_b1,[m,1])
    grad_b2 = np.reshape(grad_b2,[10,1])
    return grad_W1, grad_W2, grad_b1, grad_b2


def ComputeAccuracy(P, input_label_no_onehot, batch_size):
    Q = P.argmax(axis=0)  # Predict label
    #print(Q)
    Y = input_label_no_onehot
    diff = Q - Y
    acc = np.sum(diff == 0)/batch_size
    return acc

def Compute_momentum(grad_W1, grad_W2, grad_b1, grad_b2, v_b1, v_b2, v_W1, v_W2):
    v_b1 = rho * v_b1 + learning_rate * grad_b1
    v_b2 = rho * v_b2 + learning_rate * grad_b2
    v_W1 = rho * v_W1 + learning_rate * grad_W1
    v_W2 = rho * v_W2 + learning_rate * grad_W2
    return v_b1, v_b2, v_W1, v_W2


#Parameter
batch_size = 100
lam = 0.004
rho = 0.9
m = 50 #number of hidden nodes
MAX = 20
learning_rate = 0.028
decay_rate = 0.9
training_data = 49000

#Load data and initialization

[data_1, label_1, data_length_1, label_no_onehot_1] = LoadBatch("data_batch_1.mat")
[data_2, label_2, data_length_2, label_no_onehot_2] = LoadBatch("data_batch_2.mat")
[data_3, label_3, data_length_3, label_no_onehot_3] = LoadBatch("data_batch_3.mat")
[data_4, label_4, data_length_4, label_no_onehot_4] = LoadBatch("data_batch_4.mat")
[data_5, label_5, data_length_5, label_no_onehot_5] = LoadBatch("data_batch_5.mat")

[data_test, label_test, data_length_test, label_no_onehot_test] = LoadBatch("test_batch.mat")
data_1_mean = np.mean(data_1,1)
data_1_mean = np.reshape(data_1_mean,[3072,1])
data_1 = data_1 - data_1_mean
data_2 = data_2 - data_1_mean
data_3 = data_3 - data_1_mean
data_4 = data_4 - data_1_mean
data_5 = data_5 - data_1_mean
data_test = data_test - data_1_mean


data_real = np.concatenate((data_1,data_2,data_3,data_4,data_5[:,0:9000]),axis=1)
label_real = np.concatenate((label_1,label_2,label_3,label_4,label_5[:,0:9000]),axis=1)
data_valid = data_5[:,9000:10000]
label_valid = label_5[:,9000:10000]
label_no_onehot_real=np.concatenate((label_no_onehot_1,label_no_onehot_2,label_no_onehot_3,label_no_onehot_4,label_no_onehot_5[0:9000]),axis=0)
label_no_onehot_valid=label_no_onehot_5[9000:10000]
data_length_real = 49000
data_length_valid = 1000

lr_max = 0.03
lr_min = 0.01

lam_max = 0.007
lam_min = 0.001


#Start training!

value = []

for i in range(50): #START STOP STEP
    learning_rate = np.random.uniform(lr_min, lr_max)
    lam = np.random.uniform(lam_min, lam_max)

    [W1, W2, b1, b2] = initialization(m)
    lr = learning_rate # Store the origin learning rate before weight decay.
    J_store_1 = []
    J_store_2 = []
    loss_store_1 = []
    loss_store_2 = []
    acc_1 = []
    acc_2 = []
    v_b1 = 0
    v_b2 = 0
    v_W1 = 0
    v_W2 = 0 #Initialization of momentum


    for epoch in range(MAX):
        learning_rate = learning_rate * decay_rate
        #print("This is epoch",epoch)

        #learning_rate = learning_rate * 0.9
        for i in range(int(training_data/batch_size)):
        # for i in range(1):

            input_data = data_real[:,i*batch_size:(i+1)*batch_size]
            input_label = label_real[:,i*batch_size:(i+1)*batch_size]
            input_label_no_onehot = label_no_onehot_real[i*batch_size:(i+1)*batch_size]
            s1 = Compute_S(W1, b1, input_data)
            h = Compute_h(s1)
            s2 = Compute_S(W2,b2,h)
            P = EvaluationClassifier(s2)  # 10 * Batch_size
            grad_W1, grad_W2,  grad_b1, grad_b2 = ComputeGradients(W1, W2, b1, b2, P, input_data, input_label, lam, batch_size,m,h)

            [v_b1, v_b2, v_W1, v_W2] = Compute_momentum(grad_W1, grad_W2, grad_b1, grad_b2, v_b1, v_b2, v_W1, v_W2)
            W1 = W1 - v_W1
            W2 = W2 - v_W2
            b1 = b1 - v_b1
            b2 = b2 - v_b2

        s1 = Compute_S(W1,b1,data_real)
        h = Compute_h(s1)
        s2 = Compute_S(W2, b2, h)
        P_use = EvaluationClassifier(s2)
        J,loss = ComputeCost(label_real, lam, data_length_real, P_use,W1,W2)
        acc = ComputeAccuracy(P_use,label_no_onehot_real, data_length_real)
        J_store_1.append(J)
        acc_1.append(acc)
        loss_store_1.append(loss)
    #    print("Accuracy on training data:", acc)
        # We run our model on validation set

        s1 = Compute_S(W1,b1,data_valid)
        h = Compute_h(s1)
        s2 = Compute_S(W2, b2, h)
        P_use = EvaluationClassifier(s2)
        J,loss = ComputeCost(label_valid, lam, data_length_valid, P_use,W1,W2)
        acc = ComputeAccuracy(P_use, label_no_onehot_valid, data_length_valid)
        J_store_2.append(J)
        acc_2.append(acc)
        loss_store_2.append(loss)
    #    print("Accuracy on validation set:",acc)

    s1 = Compute_S(W1,b1,data_test)
    h = Compute_h(s1)
    s2 = Compute_S(W2, b2, h)
    P_use = EvaluationClassifier(s2)
    J,loss = ComputeCost(label_test, lam, data_length_test, P_use,W1,W2)
    acc = ComputeAccuracy(P_use, label_no_onehot_test, data_length_test)
#    print("Accuracy on test set:",acc)
    print([lam,lr,acc])
    value.append([lam,lr,acc])

value = np.array(value)
np.savetxt('value.txt', value)


x_axis = range(MAX)


#W = W - np.min(W)
#W = W/np.max(W)
#pic = np.reshape(W[0,:],[32,32,3])
#dataNew = 'W.mat'
#np.savetxt('W_opti.txt',W)
#pic = pic.transpose(2,1,3)
#pic2 = np.reshape(np.floor(data_1[:,1]*255),[32,32,3])
#pic2 = pic2.transpose(1,0,2)
#plt.figure(1)
#plt.imshow(pic)
#figure
#
# plt.figure(1)
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.plot(x_axis,loss_store_1,'r',label='training data')
# plt.plot(x_axis,loss_store_2,'g',label='validation data')
# plt.legend()
# plt.savefig('loss_opti.png')
#
#
# plt.figure(2)
# plt.xlabel("epoch")
# plt.ylabel("cost")
# plt.plot(x_axis,J_store_1,'r',label='training data')
# plt.plot(x_axis,J_store_2,'g',label='validation data')
# plt.legend()
# plt.savefig('cost_opti.png')
#
# plt.figure(3)
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.plot(x_axis,acc_1,'r',label='training data')
# plt.plot(x_axis,acc_2,'g',label='validation data')
# plt.legend()
#
#
#
# plt.show()
