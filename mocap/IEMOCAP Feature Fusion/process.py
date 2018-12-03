import pickle
import sys
import pandas as pd
import numpy as np

with open('newnewdata\\split_as_in_original_paper.pickle','rb') as handle2:
    speech=pickle.load(handle2)

def accuracy(ypre,y):
    right=0
    total=ypre.shape[0]
    for i in range(total):
        if (ypre[i]-1)==y[i]:
            right+=1
    acc=right/total
    return acc

def ydata_change(ydata):
    y=np.zeros(ydata.shape[0])
    for i in range(ydata.shape[0]):
        for j in range(ydata.shape[1]):
            if ydata[i,j]!=0:
                y[i]=j
    return y

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


with open('newnewdata\\cat_feat_mfcc_100_256.pkl','rb') as f:
	speech=pickle.load(f,encoding='latin1')

with open('newnewdata\\combined_256_mfcc_VQ_lpcc.pkl','rb') as f:
	speech1=pickle.load(f,encoding='latin1')

y=speech['cat']
x=speech['feat']

train=x['train']
test=x['test']
np.savetxt('speech_train1.txt',train)
np.savetxt('speech_test1.txt',test)
