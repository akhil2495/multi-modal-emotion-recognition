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

with open('newnewdata\\original_data\\cat_feat_mfcc_100_256.pkl','rb') as f:
	speech1=pickle.load(f,encoding='latin1')

with open('newnewdata\\original_data\\combined_256_mfcc_VQ_lpcc.pkl','rb') as f1:
	speech2=pickle.load(f1,encoding='latin1')

Text70_read=pd.DataFrame(pd.read_csv('newnewdata\\original_data\\text_features_sent2vec(lstm_accu_70%)_256.txt',sep=' ',header=None))
T70=np.asarray(Text70_read.values,dtype=np.float64)

Text69_read=pd.DataFrame(pd.read_csv('newnewdata\\original_data\\text_features_skip-thought(lstm_accu_69%)_256.txt',sep=' ',header=None))
T69=np.asarray(Text69_read.values,dtype=np.float64)

Text68_read=pd.DataFrame(pd.read_csv('newnewdata\\original_data\\text_features_doc2vec(lstm_accu_68%)_256.txt',sep=' ',header=None))
T68=np.asarray(Text68_read.values,dtype=np.float64)

tr=name['train']
te=name['test']

y_speech1=speech1['cat']
x_speech1=speech1['feat']
speech1_train_x=np.array(x_speech1['train'])
speech1_test_x=np.array(x_speech1['test'])
speech1_train_y=np.array(y_speech1['train'])
speech1_test_y=np.array(y_speech1['test'])

y_speech2=speech2['cat']
x_speech2=speech2['feat']
speech2_train_x=np.array(x_speech2['train'])
speech2_test_x=np.array(x_speech2['test'])
speech2_train_y=np.array(y_speech2['train'])
speech2_test_y=np.array(y_speech2['test'])



dict_speech1={}
dict_ys1={}
dict_speech2={}
dict_ys2={}



# train_len=len(tr)
# test_len=len(te)
# for i in range(train_len):
# 	dict_speech1[tr[i]]=train[i]
# 	dict_y[tr[i]]=ytrain[i]

# for j in range(test_len):
# 	dict_speech[te[j]]=test[j]
# 	dict_y[te[j]]=ytest[j]





# train=tr
# test=te
# # speech_train=[]
# # text_train=[]
# Y_train=[]
# mocap_rot_train = []
# for ses_mod in data2:
# 	if ses_mod['id'] in train:
# # 			# text_train.append(dict_text[ses_mod['id']])
# # 			speech_train.append(dict_speech[ses_mod['id']])
# # 			y=dict_y[ses_mod['id']]-1
# 			# Y_train.append(y)
# 			Y_train.append(ses_mod['emotion'])
# 			x_rot = ses_mod['mocap_rot']
# 			if(x_rot.shape != (200,194)):
# 				x_rot = np.zeros((200,194))  
# 			x_rot[np.isnan(x_rot)]=0
# 			mocap_rot_train.append( x_rot )

# mocap_rot_train = np.array(mocap_rot_train)			
# Y_train = label_binarize(Y_train,emotions_used)
# # voice_train=np.array(voice_train)
# # text_train=np.array(text_train)

# # speech_train=np.array(speech_train)
# # y_train=np.array(Y_train)
# # np.savetxt('speech_train2.txt',speech_train)
# # np.savetxt('y_train2.txt',y_train)
# # print(speech_train.shape)
# normal_feature_train=mocap_rot_train[:,:,165:194]
# np.savetxt('normal_feature_train.txt',normal_feature_train)
# np.savetxt('y1train.txt',Y_train)

# speech_test=[]
# text_test=[]
# # voice_test=[]
# Y_test=[]
# mocap_rot_test = []
# for ses_mod in data2:
# 	if ses_mod['id'] in test:
# # 			# text_test.append(dict_text[ses_mod['id']])
# # 			speech_test.append(dict_speech[ses_mod['id']])
# # 			y=dict_y[ses_mod['id']]-1
# # 			Y_test.append(y)
# 			Y_test.append(ses_mod['emotion'])
# 			x_rot = ses_mod['mocap_rot']
# 			if(x_rot.shape != (200,194)):
# 				x_rot = np.zeros((200,194))  
# 			x_rot[np.isnan(x_rot)]=0
# 			mocap_rot_test.append( x_rot )
			

# mocap_rot_test = np.array(mocap_rot_test)			
# Y_test = label_binarize(Y_test,emotions_used)
# normal_feature_test=mocap_rot_test[:,:,165:194]
# np.savetxt('normal_feature_test.txt',normal_feature_test)
# np.savetxt('y1test.txt',Y_test)
# print(normal_feature_test.shape)
# # voice_test=np.array(voice_test)

# # text_test=np.array(text_test)
# # np.savetxt('text70_test.txt',text_test)
# # print(text_test.shape)

# speech_test=np.array(speech_test)
# y_test=np.array(Y_test)
# np.savetxt('speech_test2.txt',speech_test)
# np.savetxt('y_test2.txt',y_test)
# print(speech_test.shape)
# print(y_test.shape)



# xtrain=normal_feature_train
# xtest=normal_feature_test

#max pooling

train=tr
test=te

dict_test={}
for ses_mod in data2:
	if ses_mod['id'] in test:
		dict_test[ses_mod['id']]=ses_mod['mocap_rot']


def search(id,dict):
	if id in dict.keys():
		flag=0
		temp=dict[id]
	else:
		flag=1
		temp=np.zeros((200,194))
	return temp,flag

dflag=[]
Y_test=[]
mocap_rot_test = []
for ses_mod in data1:
	if ses_mod['id'] in test:
			# Y_test.append(ses_mod['emotion'])
			x_rot,flag= search(ses_mod['id'],dict_test)
			# dflag.append(flag)
			if(x_rot.shape != (200,194)):
				x_rot = np.zeros((200,194))  
			x_rot[np.isnan(x_rot)]=0
			mocap_rot_test.append( x_rot )
			
mocap_rot_test = np.array(mocap_rot_test)			
# Y_test = label_binarize(Y_test,emotions_used)
# dflag=np.array(dflag)
# print(dflag.shape)
# print(dflag)
normal_feature_test=mocap_rot_test[:,:,165:194]
print(normal_feature_test.shape)
# print(Y_test.shape)
# np.savetxt('y1test.txt',Y_test)
