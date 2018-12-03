import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle

def ydata_change(ydata):
    y=np.zeros(ydata.shape[0])
    for i in range(ydata.shape[0]):
        for j in range(ydata.shape[1]):
            if ydata[i,j]!=0:
                y[i]=j
    return y
   
def select_feature(xtrain,xtest,ytrain,num):
    clf=RandomForestClassifier()
    clf.fit(xtrain,ytrain)
    weight=clf.feature_importances_
    newtrain=select(weight,xtrain,num)
    newtest=select(weight,xtest,num)
    return newtrain,newtest

def select(weight,x,num):
    c=part(weight,num)
    output=np.zeros((x.shape[0],num))
    for j in range(x.shape[0]):
        for i in range(num):
            output[j,i]=x[j,c[i]]
    
    return output

def part(weight,num):
    b=np.zeros((weight.shape[0],2))
    b1=np.zeros((weight.shape[0],2))
    for i in range(weight.shape[0]):
        b[i,0]=weight[i]
        b[i,1]=i

    b1=np.array([np.lexsort(-b[:,::-1].T)]).reshape(weight.shape[0])

    return b1

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
xtrain=np.c_[text68_xtrain,cnn_xtrain,lstm_xtrain,speech_xtrain1]
xtest=np.c_[text68_xtest,cnn_xtest,lstm_xtest,speech_xtest1]


newtrain,newtest=select_feature(xtrain,xtest,ytrain,256)
np.savetxt('train.txt',newtrain)
print(newtrain.shape)
np.savetxt('test.txt',newtest)
print(newtest.shape)
