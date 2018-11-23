#from python_speech_features import mfcc as mfc
from python_speech_features import sigproc
import python_speech_features
from audiolazy.lazy_lpc import lpc
import scipy.io.wavfile as wav
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import time
import librosa
import subprocess

class speech_feat_extract():
    
    def __init__(self, sessions, datapath, categories = ['ang', 'exc', 'neu', 'sad']):
        self.sessions = sessions
        self.datapath = datapath
        
        self.category_list = categories
        self.mfcc = {}
        self.zcr = {}
        self.pitch = {}
        self.mfcc_stat = {}
        self.zcr_stat = {}
        self.pitch_stat = {}
        
        self.id2label = {}
        self.label2ids = {}

        def parse_groundtruth(fpath):
            with open(fpath) as fil:
	        for line in fil:
                    if line.startswith('['):
                        words = line.rstrip().split('\t')
                        self.id2label[words[1]] = words[2]
                        if self.label2ids.has_key(words[2]):
                            self.label2ids[words[2]].append(words[1])
                        else:
                            self.label2ids[words[2]] = [words[1]]
        
        for session in sessions:
            for subdir, dirs, files in os.walk(datapath + session + '/dialog/EmoEvaluation/all'):
                for f in files:
                    fpath = subdir + os.sep + f
                    if fpath.endswith('.txt'):
                        parse_groundtruth(fpath)
                        
        self.cat2catid = {}
        for i in range(len(categories)):
            self.cat2catid[categories[i]] = i+1
            
        self.wavpaths = self.filepath_helper('/sentences/wav', '.wav')
        print len(self.wavpaths)

    def pad_sequence_into_array(self, Xs, maxlen=200, truncating='post', padding='post', value=0.):

    	Nsamples = len(Xs)
    	if maxlen is None:
            lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
            maxlen = np.max(lengths)

        Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
        Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
        for i in range(Nsamples):
            x = Xs[i]
            if truncating == 'pre':
                trunc = x[-maxlen:]
            elif truncating == 'post':
                trunc = x[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % truncating)
            if padding == 'post':
                Xout[i, :len(trunc)] = trunc
                Mask[i, :len(trunc)] = 1
            elif padding == 'pre':
                Xout[i, -len(trunc):] = trunc
                Mask[i, -len(trunc):] = 1
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
        return Xout, Mask
            
    def filepath_helper(self, relpath, filetype):
        filepaths = []
        for session in self.sessions:
            for subdir, dirs, files in os.walk(self.datapath + session + relpath):
                for f in files:
                    if f.endswith(filetype) and self.id2label[f[:-4]] in self.category_list:
                        filepaths.append([subdir + os.sep + f, f[:-4]])
        return filepaths                        
        
    def compute_mfcc(self, pad, save=False):
        wavpath = '/sentences/wav/'
        filepaths = self.wavpaths
        print 'Computing MFCC for ' + str(len(filepaths)) + ' utterances'
        i = 0
        for f in filepaths:
            sig = librosa.load(f[0])
            #(rate,sig) = wav.read(f[0])
            mfcc_feat = librosa.feature.mfcc(sig[0], n_mfcc=14)
            m = mfcc_feat
            mfcc_feat = np.concatenate((mfcc_feat, librosa.feature.delta(mfcc_feat, order = 1)), axis = 0)
            mfcc_feat = np.concatenate((mfcc_feat, librosa.feature.delta(m, order = 2)), axis = 0)
            if i%100 == 0:
                print i
                print mfcc_feat.shape
            i += 1
            self.mfcc[f[1]] = self.pad_sequence_into_array(mfcc_feat, maxlen=pad)
            #spectral_centroid = logfbank(sig,rate)
            #self.mfcc[f[1]] = np.concatenate((self.mfcc[f[1]], log_filt_bank), axis = 1)
            #self.mfcc[f[1]] = np.concatenate((self.mfcc[f[1]], librosa.feature.delta(log_filt_bank, 1)), axis = 1)
            #self.mfcc[f[1]] = np.concatenate((self.mfcc[f[1]], librosa.feature.delta(log_filt_bank, 2)), axis = 1)

        if save:
            with open('feat/mfcc.pkl', 'wb') as f:
                pickle.dump(self.mfcc, f)
        
    def compute_zcr(self, pad, save=False):
        filepaths = self.wavpaths
        for f in filepaths:
            sig, sr = librosa.load(f[0])
            zcr_rate = librosa.feature.zero_crossing_rate(sig)[0]
            zcr_rate = np.reshape(zcr_rate, (-1,zcr_rate.shape[0]))
            A = librosa.feature.delta(zcr_rate, order = 1)
            print A.shape
            m = np.concatenate((zcr_rate, A), axis = 0)
            A = librosa.feature.delta(zcr_rate, order = 2)
            self.zcr[f[1]] = self.pad_sequence_into_array(np.concatenate((m, A), axis = 0), maxlen=pad)
        if save:
            with open('feat/zcr.pkl', 'wb') as f:
                pickle.dump(self.zcr, f)
        
    def compute_pitch(self, pad, save=False):
        filepaths = self.wavpaths
        for f in filepaths:
            subprocess.call(['/usr/bin/praat', '--run', 'extract_pitch.praat', f[0]])
            pitch = []#{}
            with open('temp.pitch') as fil:
                for i in fil:
                    val = i.split()
                    if val[1] != '--undefined--' and val[0] != 'Pitch':
                        #pitch[float(val[0])] = float(val[1])
                        pitch.append(float(val[1]))
            temp = np.reshape(np.array(pitch), (1, -1))
            print temp.shape
            self.pitch[f[1]] = self.pad_sequence_into_array(temp, maxlen=pad)
        if save:
            with open('feat/pitch.pkl', 'wb') as f:
                pickle.dump(self.pitch, f)

    def compute_chroma_cens(self, pad, save=False):
        filepaths = self.wavpaths
        self.chroma_cens = {}
        for f in filepaths:
            sig, sr = librosa.load(f[0])
            self.chroma_cens[f[1]] = self.pad_sequence_into_array(librosa.feature.chroma_cens(sig, n_chroma=12), maxlen=pad)
        if save:
            with open('feat/chroma_cens.pkl', 'wb') as f:
                pickle.dump(self.chroma_cens, f)

    def compute_jitter(self, save=False):
        # obsolete
        if self.pitch:
            self.jitter = {}
            for k,v in self.pitch.items():
                self.jitter[k] = sum([abs(v[i]-v[i+1]) for i in range(len(v)-1)])
        if save:
            with open('feat/jitter.pkl', 'wb') as f:
                pickle.dump(self.jitter, f)

    def compute_voiceQuality(self, save=False):
        filepaths = self.wavpaths
        self.voiceQuality = {}
        j = 0
        for f in filepaths:
            subprocess.call(['/usr/bin/praat', '--run', 'extract_voiceQuality.praat', f[0]])
            with open('temp.voiceQuality') as fil:
                val = []
                for i in fil:
                    val += [float(k) if k!='--undefined--' else 0 for k in i.rstrip().split(' ')]
                print val
                print len(val)
                if j%100 == 0:
                    print j
                j += 1
            self.voiceQuality[f[1]] = val
        if save:
            with open('feat/voiceQuality.pkl', 'wb') as f:
                pickle.dump(self.voiceQuality, f)

    def compute_lpcc(self, pad, save=False):
        filepaths = self.wavpaths
        self.lpcc = {}
        self.sr = {}
        self.frames = {}
        st = time.time()
        for f in filepaths:
            sig, self.sr[f[1]] = librosa.load(f[0])
            self.frames[f[1]] = sigproc.framesig(sig, frame_len=2200, frame_step=1100)
            temp = list(self.frames[f[1]])
            arr = np.array([np.array(lpc.kautocor(i, 12).numerator) for i in temp])
            print arr.shape
            self.lpcc[f[1]] = self.pad_sequence_into_array(arr.transpose(), maxlen=pad)
        if save:
            with open('feat/lpcc.pkl', 'wb') as f:
                pickle.dump(self.lpcc, f)

    def compute_residual(self, pad, save=False):
        filepaths = self.wavpaths
        frames = {}
        fno = 0
        for f in filepaths:
            if fno%100 == 0:
                print fno
            fno += 1
            sig, sr = librosa.load(f[0])
            frames[f[1]] = sigproc.framesig(sig, frame_len=2200, frame_step=1100)
        print 'Completed frames'
        self.residual = {}
        for k,v in frames.items():
            residual = []
            for frame in list(v):
                analysis_filt = lpc.kautocor(frame, 12)
                residual.append(np.array(list(analysis_filt(frame))))
            residual, _ = self.pad_sequence_into_array(np.array(residual).transpose(), maxlen=pad)
            print residual.shape
            self.residual[k] = np.array(residual)
            

    def compute_rmfcc(self, pad, save=False):
        filepaths = self.wavpaths
        frames = {}
        fno = 0
        self.af = {}
        self.rmfcc = {}
        for f in filepaths:
            if fno%10 == 0:
                print fno
            fno += 1
            sig, sr = librosa.load(f[0])
            frames = sigproc.framesig(sig, frame_len=500, frame_step=250)
            af = []
            rmfcc = []
            for frame in frames:
                analysis_filt = lpc.autocor(frame, 12)
                af.append(analysis_filt)
                residual = np.array(list(analysis_filt(frame)))
                temp = list(python_speech_features.mfcc(frame, sr, winlen=0.022))
                rmfcc.append(temp[0])
            rmfcc,_ = self.pad_sequence_into_array(np.array(rmfcc).transpose(), maxlen=pad)
            print rmfcc.shape
            self.rmfcc[f[1]] = rmfcc
            self.af[f[1]] = af
        with open('analysis_filt.pkl', 'wb') as f:
            pickle.dump(self.af, f)


    def compute_stat(self, feat, save=False):
        if feat == 'mfcc':
            if self.mfcc:
                X = self.mfcc
            else:
                self.compute_mfcc()
                X = self.mfcc
        elif feat == 'zcr':
            if self.zcr:
                X = self.zcr
            else:
                self.compute_zcr()
                X = self.zcr
        
        X_mean = np.empty((0,60), float)
        X_var = np.empty((0,60), float)
        X_min = np.empty((0,60), float)
        X_max = np.empty((0,60), float)
        label = []
        fnames = []
        for k,v in X.items():
            A = np.mean(v, axis=0)
            X_mean = np.concatenate((X_mean, np.reshape(A, (-1, A.shape[0]))), axis=0)
            A = np.var(v, axis=0)
            X_var = np.concatenate((X_var, np.reshape(A, (-1, A.shape[0]))), axis=0)
            A = np.min(v, axis=0)
            X_min = np.concatenate((X_min, np.reshape(A, (-1, A.shape[0]))), axis=0)
            A = np.max(v, axis=0)
            X_max = np.concatenate((X_max, np.reshape(A, (-1, A.shape[0]))), axis=0)
            label.append(self.cat2catid[self.id2label[k]])
            fnames.append(k)
                
        if feat == 'mfcc':
            mfcc_stat = np.concatenate((X_mean, X_var), axis = 1)
            mfcc_stat = np.concatenate((mfcc_stat, X_min), axis = 1)
            self.mfcc_stat = np.concatenate((mfcc_stat, X_max), axis = 1)
            self.label = label
            self.fnames = fnames
            if save:
                with open('feat/mfcc_stat.pkl', 'wb') as f:
                    pickle.dump(self.mfcc_stat, f)
        elif feat == 'zcr':
            zcr_stat = np.concatenate((X_mean, X_var), axis=1)
            zcr_stat = np.concatenate((zcr_stat, X_min), axis = 1)
            self.zcr_stat = np.concatenate((zcr_stat, X_max), axis = 1)
            self.label = label
            if save:
                with open('feat/zcr_stat.pkl', 'wb') as f:
                    pickle.dump(self.zcr_stat, f)
