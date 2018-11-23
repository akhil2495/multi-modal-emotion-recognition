from features import speech_feat_extract
import numpy as np
import pickle

with open('split.pkl') as f:
	label = pickle.load(f)

sess = ['Session'+str(i+1) for i in range(5)]
data = speech_feat_extract(sess, '../../../../HBA/IEMOCAP_full_release/')

# mfcc extraction
def mfcc_extract(data, pad):
	data.compute_mfcc(pad)
	train = []
	test = []
	lab_tr = []
	lab_te = []
	for k in label['train']:
		v = data.mfcc[k]
		train.append(v[0].transpose())
		lab_tr.append(data.cat2catid[data.id2label[k]])
	for k in label['test']:
		v = data.mfcc[k]
		test.append(v[0].transpose())
		lab_te.append(data.cat2catid[data.id2label[k]])
	temp = {'feat':{'train':np.array(train), 'test':np.array(test)}, 
        	'cat':{'train':np.array(lab_tr), 'test':np.array(lab_te)}}
	print np.array(train).shape
	print np.array(lab_tr).shape
	with open('feat/80_20/mfcc_' + str(pad) + '.pkl', 'wb') as f:
		pickle.dump(temp, f)

# zcr extraction
def zcr_extract(data, pad):
	data.compute_zcr(pad)
	train = []
	test = []
	lab_tr = []
	lab_te = []
	for k in label['train']:
		v = data.zcr[k]
		train.append(v[0].transpose())
		lab_tr.append(data.cat2catid[data.id2label[k]])
	for k in label['test']:
		v = data.zcr[k]
		test.append(v[0].transpose())
		lab_te.append(data.cat2catid[data.id2label[k]])
	temp = {'feat':{'train':np.array(train), 'test':np.array(test)},
            'cat':{'train':np.array(lab_tr), 'test':np.array(lab_te)}}
	print np.array(train).shape
	print np.array(lab_tr).shape
	with open('feat/80_20/zcr_' + str(pad) + '.pkl', 'wb') as f:
		pickle.dump(temp, f)

# pitch extraction
def pitch_extract(data, pad):
	data.compute_pitch(pad)
	train = []
	test = []
	lab_tr = []
	lab_te = []
	for k in label['train']:
		v = data.pitch[k]
		train.append(v[0].transpose())
		lab_tr.append(data.cat2catid[data.id2label[k]])
	for k in label['test']:
		v = data.pitch[k]
		test.append(v[0].transpose())
		lab_te.append(data.cat2catid[data.id2label[k]])
	temp = {'feat':{'train':np.array(train), 'test':np.array(test)},
            'cat':{'train':np.array(lab_tr), 'test':np.array(lab_te)}}
	print np.array(train).shape
	print np.array(lab_tr).shape
	with open('feat/80_20/pitch_' + str(pad) + '.pkl', 'wb') as f:
		pickle.dump(temp, f)

# chroma extraction
def chroma_extract(data, pad):
	data.compute_chroma_cens(pad)
	train = []
	test = []
	lab_tr = []
	lab_te = []
	for k,v in data.chroma_cens.items():
		if k in label['train']:
			train.append(v[0].transpose())
			lab_tr.append(data.cat2catid[data.id2label[k]])
		else:
			test.append(v[0].transpose())
			lab_te.append(data.cat2catid[data.id2label[k]])
	temp = {'feat':{'train':np.array(train), 'test':np.array(test)},
            'cat':{'train':np.array(lab_tr), 'test':np.array(lab_te)}}
	print np.array(train).shape
	print np.array(lab_tr).shape
	with open('feat/80_20/chroma_cens_' + str(pad) + '.pkl', 'wb') as f:
		pickle.dump(temp, f)

def lpcc_extract(data, pad):
	data.compute_lpcc(pad)
	with open('feat/lpcc.pkl', 'wb') as f:
		pickle.dump(data.lpcc, f)
	train = []
	test = []
	lab_tr = []
	lab_te = []
	for k in label['train']:
		v = data.lpcc[k]
		train.append(v[0].transpose())
		lab_tr.append(data.cat2catid[data.id2label[k]])
	for k in label['test']:
		v = data.lpcc[k]
		test.append(v[0].transpose())
		lab_te.append(data.cat2catid[data.id2label[k]])
	temp = {'feat':{'train':np.array(train), 'test':np.array(test)},
            'cat':{'train':np.array(lab_tr), 'test':np.array(lab_te)}}
	print np.array(train).shape
	print np.array(lab_tr).shape
	with open('feat/80_20/lpcc_' + str(pad) + '.pkl', 'wb') as f:
		pickle.dump(temp, f)

def VQ_extract(data):
	data.compute_voiceQuality()
	with open('feat/voiceQuality.pkl', 'wb') as f:
		pickle.dump(data.voiceQuality, f)
	train = []
	test = []
	lab_tr = []
	lab_te = []
	for k in label['train']:
		v = data.voiceQuality[k]
		train.append(v)
		lab_tr.append(data.cat2catid[data.id2label[k]])
	for k in label['test']:
		v = data.voiceQuality[k]
		test.append(v)
		lab_te.append(data.cat2catid[data.id2label[k]])
	temp = {'feat':{'train':np.array(train), 'test':np.array(test)},
            'cat':{'train':np.array(lab_tr), 'test':np.array(lab_te)}}
	print np.array(train).shape
	print np.array(lab_tr).shape
	with open('feat/80_20/voiceQuality.pkl', 'wb') as f:
		pickle.dump(temp, f)

def residual_extract(data, pad):
	data.compute_residual(pad)
	with open('feat/residual.pkl', 'wb') as f:
		pickle.dump(data.residual, f)
	train = []
	test = []
	lab_tr = []
	lab_te = []
	for k in label['train']:
		v = data.residual[k]
		train.append(v[0].transpose())
		lab_tr.append(data.cat2catid[data.id2label[k]])
	for k in label['test']:
		v = data.residual[k]
		test.append(v[0].transpose())
		lab_te.append(data.cat2catid[data.id2label[k]])
	temp = {'feat':{'train':np.array(train), 'test':np.array(test)},
            'cat':{'train':np.array(lab_tr), 'test':np.array(lab_te)}}
	print np.array(train).shape
	print np.array(lab_tr).shape
	with open('feat/80_20/residual_' + str(pad) + '.pkl', 'wb') as f:
		pickle.dump(temp, f)

def rmfcc_extract(data, pad):
	data.compute_rmfcc(pad)
	with open('feat/rmfcc.pkl', 'wb') as f:
		pickle.dump(data.rmfcc, f)
	train = []
	test = []
	lab_tr = []
	lab_te = []
	for k in label['train']:
		v = data.rmfcc[k]
		train.append(v[0].transpose())
		lab_tr.append(data.cat2catid[data.id2label[k]])
	for k in label['test']:
		v = data.rmfcc[k]
		test.append(v[0].transpose())
		lab_te.append(data.cat2catid[data.id2label[k]])
	temp = {'feat':{'train':np.array(train), 'test':np.array(test)},
            'cat':{'train':np.array(lab_tr), 'test':np.array(lab_te)}}
	print np.array(train).shape
	print np.array(lab_tr).shape
	with open('feat/80_20/rmfcc_' + str(pad) + '.pkl', 'wb') as f:
		pickle.dump(temp, f)

#mfcc_extract(data, 100)
#zcr_extract(data, 100)
#pitch_extract(data, 40)
#chroma_extract(data, 100)

#mfcc_extract(data, 150)
#zcr_extract(data, 150)
#pitch_extract(data, 30)
#chroma_extract(data, 150)

#mfcc_extract(data, 200)
#zcr_extract(data, 200)
#pitch_extract(data, 50)
#chroma_extract(data, 200)

#VQ_extract(data)
#lpcc_extract(data, 80)

#residual_extract(data, 80)
rmfcc_extract(data, 400)
