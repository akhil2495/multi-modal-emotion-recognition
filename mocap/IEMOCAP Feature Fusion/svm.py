import pandas as pd 
import numpy as np 
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle 

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
        if (ypre[i])==y[i]:
            right+=1
    acc=right/total
    return acc

#prepare for all the data
#cnn data
XRead_test_CNN=pd.DataFrame(pd.read_csv('newnewdata\\Adam_output_test.txt',sep=' ',header=None))
XRead_train_CNN=pd.DataFrame(pd.read_csv('newnewdata\\Adam_output_train.txt',sep=' ',header=None))
YRead_test_CNN=pd.DataFrame(pd.read_csv('newnewdata\\cnn_ytest.txt',sep=' ',header=None))
YRead_train_CNN=pd.DataFrame(pd.read_csv('newnewdata\\cnn_ytrain.txt',sep=' ',header=None))

cnn_xtrain=np.asarray(XRead_train_CNN.values,dtype=np.float64)
cnn_xtest=np.asarray(XRead_test_CNN.values,dtype=np.float64)
cnn_ytrain_ori=np.asarray(YRead_train_CNN.values,dtype=np.float64)
cnn_ytrain=ydata_change(cnn_ytrain_ori)
cnn_ytest_ori=np.asarray(YRead_test_CNN.values,dtype=np.float64)
cnn_ytest=ydata_change(cnn_ytest_ori)

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

# #speech data
# XRead_test_VOI1=pd.DataFrame(pd.read_csv('newnewdata\\speech_test.txt',sep=' ',header=None))
# XRead_train_VOI1=pd.DataFrame(pd.read_csv('newnewdata\\speech_train.txt',sep=' ',header=None))
# speech_xtrain1=np.asarray(XRead_train_VOI1.values,dtype=np.float64)
# speech_xtest1=np.asarray(XRead_test_VOI1.values,dtype=np.float64)

XRead_test_VOI2=pd.DataFrame(pd.read_csv('newnewdata\\speech_test2.txt',sep=' ',header=None))
XRead_train_VOI2=pd.DataFrame(pd.read_csv('newnewdata\\speech_train2.txt',sep=' ',header=None))
speech_xtrain2=np.asarray(XRead_train_VOI2.values,dtype=np.float64)
speech_xtest2=np.asarray(XRead_test_VOI2.values,dtype=np.float64)

# #
YRead_test_speech=pd.DataFrame(pd.read_csv('newnewdata\\y1test.txt',sep=' ',header=None))
YRead_train_speech=pd.DataFrame(pd.read_csv('newnewdata\\y1train.txt',sep=' ',header=None))
yte1=ydata_change(np.asarray(YRead_test_speech.values,dtype=np.float64))
ytr1=ydata_change(np.asarray(YRead_train_speech.values,dtype=np.float64))



#label data
ytrain=lstm_ytrain
ytest=lstm_ytest
# print(yte1.shape)
# print(ytr1.shape)
print(accuracy(yte1,ytest))
print(accuracy(ytr1,ytrain))
#svm train

# tcl_xtrain=np.c_[text70_xtrain,cnn_xtrain,lstm_xtrain]
# tcl_xtest=np.c_[text70_xtest,cnn_xtest,lstm_xtest]


#C+S




####68
# t68c_xtrain=np.c_[text68_xtrain,cnn_xtrain]
# t68c_xtest=np.c_[text68_xtest,cnn_xtest]

# t68l_xtrain=np.c_[text68_xtrain,lstm_xtrain]
# t68l_xtest=np.c_[text68_xtest,lstm_xtest]

# ####69
# t69c_xtrain=np.c_[text69_xtrain,cnn_xtrain]
# t69c_xtest=np.c_[text69_xtest,cnn_xtest]

# t69l_xtrain=np.c_[text69_xtrain,lstm_xtrain]
# t69l_xtest=np.c_[text69_xtest,lstm_xtest]

# ####70
# t70c_xtrain=np.c_[text70_xtrain,cnn_xtrain]
# t70c_xtest=np.c_[text70_xtest,cnn_xtest]

# t70l_xtrain=np.c_[text70_xtrain,lstm_xtrain]
# t70l_xtest=np.c_[text70_xtest,lstm_xtest]


# cl_xtrain=np.c_[cnn_xtrain,lstm_xtrain]
# cl_xtest=np.c_[cnn_xtest,lstm_xtest]


# tcv_xtrain=np.c_[text_xtrain,cnn_xtrain,voice_xtrain]
# tcv_xtest=np.c_[text_xtest,cnn_xtest,voice_xtest]


# #use "one against one" method
# #cnn-svm
# clfc=svm.SVC(decision_function_shape='ovo')
# clfc.fit(cnn_xtrain,ytrain)
# y_predict_train=clfc.predict(cnn_xtrain)
# y_predict_test=clfc.predict(cnn_xtest)

# acc_train=accuracy(y_predict_train,ytrain)
# acc_test=accuracy(y_predict_test,ytest)

# print('train_accuracy of CNN-SVM is: %f'%acc_train)
# print('test_accuracy of CNN-SVM is: %f'%acc_test)


# #lstm-svm
# clfl=svm.SVC(decision_function_shape='ovo')
# clfl.fit(lstm_xtrain,ytrain)
# y_predict_train=clfl.predict(lstm_xtrain)
# y_predict_test=clfl.predict(lstm_xtest)

# acc_train=accuracy(y_predict_train,ytrain)
# acc_test=accuracy(y_predict_test,ytest)

# print('train_accuracy of LSTM-SVM is: %f'%acc_train)
# print('test_accuracy of LSTM-SVM is: %f'%acc_test)

# tcl
# clftcl=svm.SVC(decision_function_shape='ovo')
# clftcl.fit(tcl_xtrain,ytrain)
# ytcl_predict_train=clftcl.predict(tcl_xtrain)
# ytcl_predict_test=clftcl.predict(tcl_xtest)

# acc_train_tcl=accuracy(ytcl_predict_train,ytrain)
# acc_test_tcl=accuracy(ytcl_predict_test,ytest)

# print('train_accuracy of Text+Cnn+Lstm-SVM is: %f'%acc_train_tcl)
# print('test_accuracy of Text+Cnn+Lstm-SVM is: %f'%acc_test_tcl)

# ###68
# tc
# clft8c=svm.SVC(decision_function_shape='ovo')
# clft8c.fit(t68c_xtrain,ytrain)
# ytc_predict_train=clft8c.predict(t68c_xtrain)
# ytc_predict_test=clft8c.predict(t68c_xtest)

# acc_train_tc=accuracy(ytc_predict_train,ytrain)
# acc_test_tc=accuracy(ytc_predict_test,ytest)

# print('train_accuracy of Text68+Cnn-SVM is: %f'%acc_train_tc)
# print('test_accuracy of Text68+Cnn-SVM is: %f'%acc_test_tc)

# # tl
# clft8l=svm.SVC(decision_function_shape='ovo')
# clft8l.fit(t68l_xtrain,ytrain)
# ytl_predict_train=clft8l.predict(t68l_xtrain)
# ytl_predict_test=clft8l.predict(t68l_xtest)

# acc_train_tl=accuracy(ytl_predict_train,ytrain)
# acc_test_tl=accuracy(ytl_predict_test,ytest)

# print('train_accuracy of Text68+Lstm-SVM is: %f'%acc_train_tl)
# print('test_accuracy of Text68+Lstm-SVM is: %f'%acc_test_tl)



# ###69
# # tc
# clft9c=svm.SVC(decision_function_shape='ovo')
# clft9c.fit(t69c_xtrain,ytrain)
# ytc_predict_train=clft9c.predict(t69c_xtrain)
# ytc_predict_test=clft9c.predict(t69c_xtest)

# acc_train_tc=accuracy(ytc_predict_train,ytrain)
# acc_test_tc=accuracy(ytc_predict_test,ytest)

# print('train_accuracy of Text69+Cnn-SVM is: %f'%acc_train_tc)
# print('test_accuracy of Text69+Cnn-SVM is: %f'%acc_test_tc)

# # tl
# clft9l=svm.SVC(decision_function_shape='ovo')
# clft9l.fit(t69l_xtrain,ytrain)
# ytl_predict_train=clft9l.predict(t69l_xtrain)
# ytl_predict_test=clft9l.predict(t69l_xtest)

# acc_train_tl=accuracy(ytl_predict_train,ytrain)
# acc_test_tl=accuracy(ytl_predict_test,ytest)

# print('train_accuracy of Text69+Lstm-SVM is: %f'%acc_train_tl)
# print('test_accuracy of Text69+Lstm-SVM is: %f'%acc_test_tl)


# ###70
# # tc
# clft0c=svm.SVC(decision_function_shape='ovo')
# clft0c.fit(t70c_xtrain,ytrain)
# ytc_predict_train=clft0c.predict(t70c_xtrain)
# ytc_predict_test=clft0c.predict(t70c_xtest)

# acc_train_tc=accuracy(ytc_predict_train,ytrain)
# acc_test_tc=accuracy(ytc_predict_test,ytest)

# print('train_accuracy of Text70+Cnn-SVM is: %f'%acc_train_tc)
# print('test_accuracy of Text70+Cnn-SVM is: %f'%acc_test_tc)

# # tl
# clft0l=svm.SVC(decision_function_shape='ovo')
# clft0l.fit(t70l_xtrain,ytrain)
# ytl_predict_train=clft0l.predict(t70l_xtrain)
# ytl_predict_test=clft0l.predict(t70l_xtest)

# acc_train_tl=accuracy(ytl_predict_train,ytrain)
# acc_test_tl=accuracy(ytl_predict_test,ytest)

# print('train_accuracy of Text70+Lstm-SVM is: %f'%acc_train_tl)
# print('test_accuracy of Text70+Lstm-SVM is: %f'%acc_test_tl)

# # cl
# clfcl=svm.SVC(decision_function_shape='ovo')
# clfcl.fit(cl_xtrain,ytrain)
# ycl_predict_train=clfcl.predict(cl_xtrain)
# ycl_predict_test=clfcl.predict(cl_xtest)

# acc_train_cl=accuracy(ycl_predict_train,ytrain)
# acc_test_cl=accuracy(ycl_predict_test,ytest)

# print('train_accuracy of Cnn+Lstm-SVM is: %f'%acc_train_cl)
# print('test_accuracy of Cnn+Lstm-SVM is: %f'%acc_test_cl)

# #t
# clft=svm.SVC(decision_function_shape='ovo')
# clft.fit(text_xtrain,ytrain)
# yt_predict_train=clft.predict(text_xtrain)
# yt_predict_test=clft.predict(text_xtest)

# acc_train_t=accuracy(yt_predict_train,ytrain)
# acc_test_t=accuracy(yt_predict_test,ytest)

# print('train_accuracy of TEXT-SVM is: %f'%acc_train_t)
# print('test_accuracy of TEXT-SVM is: %f'%acc_test_t)

# #tcv-svm
# clftcv=svm.SVC(decision_function_shape='ovo')
# clftcv.fit(tcv_xtrain,ytrain)
# y_predict_train=clftcv.predict(tcv_xtrain)
# y_predict_test=clftcv.predict(tcv_xtest)

# acc_train=accuracy(y_predict_train,ytrain)
# acc_test=accuracy(y_predict_test,ytest)

# print('train_accuracy of Text+Cnn+Voice-SVM is: %f'%acc_train)
# print('test_accuracy of Text+Cnn+Voice-SVM is: %f'%acc_test)