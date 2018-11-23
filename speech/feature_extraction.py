from features import speech_feat_extract
from sklearn.preprocessing import label_binarize
from model import LSTM_AE, LSTM_dim, LSTM_cat
import numpy as np
import pickle

class LowLevelFeat():
	def __init__():
		with open('split.pkl') as f:
			self.label = pickle.load(f)
		sess = ['Session'+str(i+1) for i in range(5)]
		self.data = speech_feat_extract(sess, '../../../../HBA/IEMOCAP_full_release/')

	def feat_extract(arg, pad=100):
		if arg == 'mfcc':
			data.compute_mfcc(pad)
			feat = data.mfcc
		elif arg == 'zcr':
			data.compute_zcr(pad)
			feat = data.zcr
		elif arg == 'pitch':
			data.compute_pitch(pad)
			feat = data.pitch
		elif arg == 'lpcc':
			data.compute_lpcc(pad)
			feat = data.lpcc
		elif arg == 'chroma':
			data.compute_chroma_cens(pad)
			feat = data.chroma_cens
		elif arg == 'VQ':
			data.compute_voiceQuality()
			feat = data.voiceQuality
		elif arg == 'rmfcc':
			data.compute_rmfcc(pad)
			feat = data.rmfcc
		train = []
		test = []
		lab_tr = []
		lab_te = []
		for k in label['train']:
			train.append(feat[k][0].transpose())
			lab_tr.append(data.cat2catid[data.id2label[k]])
		for k in label['test']:
			test.append(feat[k][0].transpose())
			lab_te.append(data.cat2catid[data.id2label[k]])
		temp = {'feat':{'train':np.array(train), 'test':np.array(test)}, 'cat':{'train':np.array(lab_tr), 'test':np.array(lab_te)}}
		print np.array(train).shape
		print np.array(lab_tr).shape
		with open('feat/80_20/' + arg + '_' + str(pad) + '.pkl', 'wb') as f:
			pickle.dump(temp, f)


class HighLevelFeat():
	def __init__(fi):
		with open('feat/80_20/' + fi[0] + '_' + fi[1] + '.pkl') as f:
			self.data = pickle.load(f)

	def extract_feat(model_type):
		tr_X = data['feat']['train']
		te_X = data['feat']['test']
		tr_y = data['feat']['train']
		te_y = data['feat']['test']
		if model_type == 'AE':
			model = LSTM_AE(tr_X.shape[1], fi[1], tr_X.shape[2])
		print model.train(tr_X)
		feat_tr = model.feature(tr_X)
		feat_te = model.feature(te_X)
		with open(model_type + '_feat_' + fi[0] + '_' + str(fi[1]) + '.pkl', 'wb') as f:
			pickle.dump({'feat':{'train':feat_tr, 'test':feat_te}, 'cat':{'train':data['cat']['train'], 'test':data['cat']['test']}}, f)

llf = LowLevelFeat()
llf.feat_extract('mfcc', 100)
llf.feat_extract('lpcc', 80)
llf.feat_extract('VQ')

HighLevelFeat('mfcc_100', 256).extract_feat('cat')
HighLevelFeat('lpcc_80', 256).extract_feat('cat')
