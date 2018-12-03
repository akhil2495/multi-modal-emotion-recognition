#import traditional feature

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
emotions_used = np.array(['ang','exc','neu','sad'])

with open('newnewdata\\original_data\\data_collected.pickle','rb') as handle:
	data2 = pickle.load(handle)
with open('DATA\\data_collected.pickle','rb') as handle1:
	data1 = pickle.load(handle1)
with open('newnewdata\\original_data\\split_as_in_original_paper.pickle','rb') as handle1:
    name=pickle.load(handle1)


train=name['train']
test=name['test']

Y_train=[]
mocap_rot_train = []
for ses_mod in data2:
	if ses_mod['id'] in train:
			Y_train.append(ses_mod['emotion'])
			x_rot = ses_mod['mocap_rot']
			if(x_rot.shape != (200,194)):
				x_rot = np.zeros((200,194))  
			x_rot[np.isnan(x_rot)]=0
			x_rot=np.amax(x_rot,axis=0)
			mocap_rot_train.append( x_rot )

mocap_rot_train = np.array(mocap_rot_train)			
Y_train = label_binarize(Y_train,emotions_used)

normal_feature_train=mocap_rot_train
np.savetxt('y1train.txt',Y_train)
np.savetxt('nor_train.txt',normal_feature_train)

dict_test={}
for ses_mod in data2:
	if ses_mod['id'] in test:
		dict_test[ses_mod['id']]=ses_mod['mocap_rot']

dict_y={}
for ses_mod in data2:
	if ses_mod['id'] in test:
		dict_y[ses_mod['id']]=ses_mod['emotion']

def search(id,dict):
	if id in dict.keys():
		flag=0
		temp=dict[id]
	else:
		flag=1
		temp=np.zeros((200,194))
	return temp,flag

def search1(id,dict):
	if id in dict.keys():
		flag=0
		temp=dict[id]
	else:
		flag=1
		temp='ang'
	return temp,flag

dflag=[]
Y_test=[]
mocap_rot_test = []
for ses_mod in data1:
	if ses_mod['id'] in test:
			x_rot,flag= search(ses_mod['id'],dict_test)
			y,flag=search1(ses_mod['id'],dict_y)
			Y_test.append(y)
			dflag.append(flag)
			if(x_rot.shape != (200,194)):
				x_rot = np.zeros((200,194))  
			x_rot[np.isnan(x_rot)]=0
			x_rot=np.amax(x_rot,axis=0)
			mocap_rot_test.append( x_rot )
			
mocap_rot_test = np.array(mocap_rot_test)			
Y_test = label_binarize(Y_test,emotions_used)

dflag=np.array(dflag)
print(dflag.shape)
print(dflag)
normal_feature_test=mocap_rot_test
print(normal_feature_test.shape)
print(Y_test.shape)
np.savetxt('y1test.txt',Y_test)
np.savetxt('nor_test.txt',normal_feature_test)



#max pooling
def max(x):
    temp=x[0]
    for i in range(x.shape[0]):
        if x[i]>temp:
            temp=x[i]
    return temp

def max_pooling(x):
    output=np.zeros((x.shape[0],x.shape[2]))
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            output[i,j]=max(x[i,:,j])
    
    return output

# out_train=max_pooling(normal_feature_train)
# out_test=max_pooling(normal_feature_test)

# print(out_train.shape)
# print(out_test.shape)
# np.savetxt('try\\normal_feature_train.txt',out_train)
# np.savetxt('try\\normal_feature_test.txt',out_test)
# # np.savetxt('flag.txt',dflag)


