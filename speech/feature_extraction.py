from features import speech_feat_extract
from sklearn.preprocessing import label_binarize
from model import LSTM_AE, LSTM_dim, LSTM_cat
import numpy as np
import pickle

class LowLevelFeat():
	def __init__(self):
		with open('split.pkl') as f:
			self.label = pickle.load(f)
		sess = ['Session'+str(i+1) for i in range(5)]
		folderpath = '../../../../../Other/IEMOCAP_full_release/' # specify the relative location to the database 
		self.data = speech_feat_extract(sess, folderpath)

	def feat_extract(self, arg, pad=100):
		if arg == 'mfcc':
			self.data.compute_mfcc(pad)
			feat = self.data.mfcc
		elif arg == 'zcr':
			self.data.compute_zcr(pad)
			feat = self.data.zcr
		elif arg == 'pitch':
			self.data.compute_pitch(pad)
			feat = self.data.pitch
		elif arg == 'lpcc':
			self.data.compute_lpcc(pad)
			feat = self.data.lpcc
		elif arg == 'chroma':
			self.data.compute_chroma_cens(pad)
			feat = self.data.chroma_cens
		elif arg == 'VQ':
			self.data.compute_voiceQuality()
			feat = self.data.voiceQuality
		elif arg == 'rmfcc':
			self.data.compute_rmfcc(pad)
			feat = self.data.rmfcc
		train = []
		test = []
		lab_tr = []
		lab_te = []
		for k in label['train']:
			train.append(feat[k][0].transpose())
			lab_tr.append(self.data.cat2catid[self.data.id2label[k]])
		for k in label['test']:
			test.append(feat[k][0].transpose())
			lab_te.append(self.data.cat2catid[self.data.id2label[k]])
		temp = {'feat':{'train':np.array(train), 'test':np.array(test)}, 'cat':{'train':np.array(lab_tr), 'test':np.array(lab_te)}}
		print np.array(train).shape
		print np.array(lab_tr).shape
		with open('feat/80_20/' + arg + '_' + str(pad) + '.pkl', 'wb') as f:
			pickle.dump(temp, f)


class HighLevelFeat():
	def __init__(self, fi):
		with open('feat/80_20/' + fi[0] + '.pkl') as f:
			self.data = pickle.load(f)
		self.fi = fi

	def extract_feat(self, model_type):
		tr_X = self.data['feat']['train']
		te_X = self.data['feat']['test']
		tr_y = label_binarize(self.data['cat']['train'], list(set(self.data['cat']['train'])))
		te_y = label_binarize(self.data['cat']['test'], list(set(self.data['cat']['test'])))
		if model_type == 'AE':
			model = LSTM_AE(tr_X.shape[1], self.fi[1], tr_X.shape[2])
			print model.train(tr_X)
		elif model_type == 'cat':
			model = LSTM_cat(tr_X.shape[1], self.fi[1], tr_X.shape[2])
			print model.train(tr_X, tr_y)
		else:
			print 'model is not specified correctly'
			return
		feat_tr = model.feature(tr_X)
		feat_te = model.feature(te_X)
		with open(model_type + '_feat_' + self.fi[0] + '_' + str(self.fi[1]) + '.pkl', 'wb') as f:
			pickle.dump({'feat':{'train':feat_tr, 'test':feat_te}, 'cat':{'train':self.data['cat']['train'], 'test':self.data['cat']['test']}}, f)

#llf = LowLevelFeat()
#llf.feat_extract('mfcc', 100)
#llf.feat_extract('lpcc', 80)
#llf.feat_extract('VQ')
#llf.feat_extract('zcr', 100)

HighLevelFeat(('mfcc_100', 256)).extract_feat('cat')
#HighLevelFeat('lpcc_80', 256).extract_feat('cat')
