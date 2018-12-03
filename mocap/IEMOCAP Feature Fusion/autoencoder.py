import pandas as pd 
import numpy as np 
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle 


import keras
from keras.layers import Dense, Input
from keras.models import Model


#change the form of ydata
#for example:
#the first classification :[1 0 0 0]-->[0]
#the second classification:[0 1 0 0]-->[1]
#......
def ydata_change(ydata):
    y=np.zeros(ydata.shape[0])
    for i in range(ydata.shape[0]):
        for j in range(ydata.shape[1]):
            if ydata[i,j]!=0:
                y[i]=j
    return y

#calculate the accuracy
def accuracy(ypre,y):
    right=0
    total=ypre.shape[0]
    for i in range(total):
        if ypre[i]==y[i]:
            right+=1
    acc=right/total
    return acc

def deletey(x,flag):
    temp=np.zeros(x.shape[0])
    count=0
    for i in range(flag.shape[0]):
        if flag[i]==0:
            temp[count]=x[i]
            count+=1
    result=temp[0:count]
    return result

def deletex(x,flag):
    temp=np.zeros(x.shape)
    count=0
    for i in range(flag.shape[0]):
        if flag[i]==0:
            temp[count,:]=x[i,:]
            count+=1
    result=temp[0:count,:]
    return result
#prepare for all the data
#cnn data
XRead_test_CNN=pd.DataFrame(pd.read_csv('newnewdata\\Adam_output_test.txt',sep=' ',header=None))
XRead_train_CNN=pd.DataFrame(pd.read_csv('newnewdata\\Adam_output_train.txt',sep=' ',header=None))
cnn_xtrain=np.asarray(XRead_train_CNN.values,dtype=np.float64)
cnn_xtest=np.asarray(XRead_test_CNN.values,dtype=np.float64)


#LSTM data
XRead_test_LSTM=pd.DataFrame(pd.read_csv('newnewdata\\lstm_test.txt',sep=' ',header=None))
XRead_train_LSTM=pd.DataFrame(pd.read_csv('newnewdata\\lstm_train.txt',sep=' ',header=None))
YRead_test_LSTM=pd.DataFrame(pd.read_csv('newnewdata\\y_test.txt',sep=' ',header=None))
YRead_train_LSTM=pd.DataFrame(pd.read_csv('newnewdata\\y_train.txt',sep=' ',header=None))

lstm_xtrain=np.asarray(XRead_train_LSTM.values,dtype=np.float64)
lstm_xtest=np.asarray(XRead_test_LSTM.values,dtype=np.float64)
lstm_ytrain_ori=np.asarray(YRead_train_LSTM.values,dtype=np.float64)
lstm_ytrain=ydata_change(lstm_ytrain_ori)
lstm_ytest_ori=np.asarray(YRead_test_LSTM.values,dtype=np.float64)
lstm_ytest=ydata_change(lstm_ytest_ori)

#text_data
XRead_test_TEXT68=pd.DataFrame(pd.read_csv('newnewdata\\text68_test.txt',sep=' ',header=None))
XRead_train_TEXT68=pd.DataFrame(pd.read_csv('newnewdata\\text68_train.txt',sep=' ',header=None))
text68_xtrain=np.asarray(XRead_train_TEXT68.values,dtype=np.float64)
text68_xtest=np.asarray(XRead_test_TEXT68.values,dtype=np.float64)

XRead_test_TEXT69=pd.DataFrame(pd.read_csv('newnewdata\\text69_test.txt',sep=' ',header=None))
XRead_train_TEXT69=pd.DataFrame(pd.read_csv('newnewdata\\text69_train.txt',sep=' ',header=None))
text69_xtrain=np.asarray(XRead_train_TEXT69.values,dtype=np.float64)
text69_xtest=np.asarray(XRead_test_TEXT69.values,dtype=np.float64)

XRead_test_TEXT70=pd.DataFrame(pd.read_csv('newnewdata\\text70_test.txt',sep=' ',header=None))
XRead_train_TEXT70=pd.DataFrame(pd.read_csv('newnewdata\\text70_train.txt',sep=' ',header=None))
text70_xtrain=np.asarray(XRead_train_TEXT70.values,dtype=np.float64)
text70_xtest=np.asarray(XRead_test_TEXT70.values,dtype=np.float64)

XRead_test_VOI2=pd.DataFrame(pd.read_csv('newnewdata\\speech_test2.txt',sep=' ',header=None))
XRead_train_VOI2=pd.DataFrame(pd.read_csv('newnewdata\\speech_train2.txt',sep=' ',header=None))
speech_xtrain2=np.asarray(XRead_train_VOI2.values,dtype=np.float64)
speech_xtest2=np.asarray(XRead_test_VOI2.values,dtype=np.float64)

XRead_test_VOI1=pd.DataFrame(pd.read_csv('newnewdata\\speech_test1.txt',sep=' ',header=None))
XRead_train_VOI1=pd.DataFrame(pd.read_csv('newnewdata\\speech_train1.txt',sep=' ',header=None))
speech_xtrain1=np.asarray(XRead_train_VOI1.values,dtype=np.float64)
speech_xtest1=np.asarray(XRead_test_VOI1.values,dtype=np.float64)

XRead_nor_test=pd.DataFrame(pd.read_csv('newnewdata\\normal_feature_test.txt',sep=' ',header=None))
XRead_nor_train=pd.DataFrame(pd.read_csv('newnewdata\\normal_feature_train.txt',sep=' ',header=None))
nor_xtrain=np.asarray(XRead_nor_train.values,dtype=np.float64)
nor_xtest=np.asarray(XRead_nor_test.values,dtype=np.float64)

flag_read=pd.DataFrame(pd.read_csv('newnewdata\\flag.txt',sep=' ',header=None))
flag=np.asarray(flag_read.values,dtype=np.float64)
#label data
ytrain=lstm_ytrain
ytest=lstm_ytest


####68
t68c_xtrain=np.c_[text68_xtrain,cnn_xtrain]
t68c_xtest=np.c_[text68_xtest,cnn_xtest]

t68l_xtrain=np.c_[text68_xtrain,lstm_xtrain]
t68l_xtest=np.c_[text68_xtest,lstm_xtest]

####69
t69c_xtrain=np.c_[text69_xtrain,cnn_xtrain]
t69c_xtest=np.c_[text69_xtest,cnn_xtest]

t69l_xtrain=np.c_[text69_xtrain,lstm_xtrain]
t69l_xtest=np.c_[text69_xtest,lstm_xtest]

####70
t70c_xtrain=np.c_[text70_xtrain,cnn_xtrain]
t70c_xtest=np.c_[text70_xtest,cnn_xtest]

t70l_xtrain=np.c_[text70_xtrain,lstm_xtrain]
t70l_xtest=np.c_[text70_xtest,lstm_xtest]








x_train=np.c_[cnn_xtrain,nor_xtrain]
x_test=np.c_[cnn_xtest,nor_xtest]
inputshape=256+29

 
# this is our input placeholder
input_data = Input(shape=(inputshape,))
 
# 编码层
encoded = Dense(256, activation='relu')(input_data)
 
# 解码层
decoded = Dense(inputshape, activation='sigmoid')(encoded)
 
# 构建自编码模型
autoencoder = Model(inputs=input_data, outputs=decoded)
 
# 构建编码模型
encoder = Model(inputs=input_data, outputs=encoded)
 
# compile autoencoder
autoencoder.compile(optimizer='Adam', loss='mse')
 
# training
autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True)
 
# plotting
encoded_train = encoder.predict(x_train)
encoded_test=encoder.predict(x_test)

np.savetxt('newnewdata\\autoencoder\\cn_train.txt',encoded_train)
np.savetxt('newnewdata\\autoencoder\\cn_test.txt',encoded_test)
