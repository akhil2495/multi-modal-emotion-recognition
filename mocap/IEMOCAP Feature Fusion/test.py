import pandas as pd 
import numpy as np 
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle 
# from sklearn.ensemble import RandomForestClassifier
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

def change(y):
    y1=np.zeros((y.shape[0],4))
    for i in range(y.shape[0]):
        if y[i]==0:
            y1[i,:]=np.array([1,0,0,0])
        elif y[i]==1:
            y1[i,:]=np.array([0,1,0,0])
        elif y[i]==2:
            y1[i,:]=np.array([0,0,1,0])
        elif y[i]==3:
            y1[i,:]=np.array([0,0,0,1])
    return y1

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

def norm(data):
    mn=min(data) 
    ma=max(data) 
    result=[(float(i)-mn)/(ma-mn) for i in data]
    return np.array(result,dtype=np.float64)

def autoNorm(data):
    temp=[]
    for i in range(data.shape[0]):
        temp.append(norm(data[i,:]))
    return np.array(temp,dtype=np.float64)

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


test=pd.DataFrame(pd.read_csv('newnewdata\\rf\\test.txt',sep=' ',header=None))
train=pd.DataFrame(pd.read_csv('newnewdata\\rf\\train.txt',sep=' ',header=None))
xtrain=np.asarray(train.values,dtype=np.float64)
xtest=np.asarray(test.values,dtype=np.float64)
# xntr=xtrain1[:,165:194]
# xnte=xtest1[:,165:194]


# XRead_test=pd.DataFrame(pd.read_csv('newnewdata\\autoencoder\\cn_test.txt',sep=' ',header=None))
# XRead_train=pd.DataFrame(pd.read_csv('newnewdata\\autoencoder\\cn_train.txt',sep=' ',header=None))
# xtrain=np.asarray(XRead_train.values,dtype=np.float64)
# xtest=np.asarray(XRead_test.values,dtype=np.float64)

# xtrain=np.c_[lstm_xtrain,xntr]
# xtest=np.c_[lstm_xtest,xnte]
# # xtest=deletex(xtest,flag)
# # ytest=deletey(ytest,flag)
# # print(accuracy(yte,ytest))


clfc=svm.SVC(decision_function_shape='ovo')

clfc.fit(xtrain,ytrain)
y_predict_train=clfc.predict(xtrain)
y_predict_test=clfc.predict(xtest)

acc_train=accuracy(y_predict_train,ytrain)
acc_test=accuracy(y_predict_test,ytest)


print('train_accuracy of SVM is: %f'%acc_train)
print('test_accuracy of SVM is: %f'%acc_test)
